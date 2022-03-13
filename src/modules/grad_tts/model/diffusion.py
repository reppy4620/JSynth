import math
import numpy as np
import torch
import torch.nn as nn

from scipy.integrate import solve_ivp
from einops import rearrange


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()
        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mlp = torch.nn.Sequential(
            nn.GELU(),
            torch.nn.Linear(time_emb_dim, dim)
        ) if time_emb_dim is not None else None
        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.conv1 = nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1)

        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask, time_emb=None):
        h = self.ds_conv(x)
        if self.mlp is not None:
            h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm(h * mask)
        h = self.conv1(h * mask)
        h = self.act(h)
        h = self.conv2(h * mask)
        output = h * mask + self.res_conv(x * mask)
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = self.scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScoreNet(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4)):
        super(ScoreNet, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList([
                    ConvNextBlock(dim_in, dim_out, time_emb_dim=dim, norm=ind!=0),
                    ConvNextBlock(dim_out, dim_out, time_emb_dim=dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else torch.nn.Identity()
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList([
                    ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                    ConvNextBlock(dim_in, dim_in, time_emb_dim=dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in)
                ])
            )
        self.final_block = ConvNextBlock(dim, dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t):
        t = self.time_mlp(t)

        x = torch.stack([mu, x], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for conv1, conv2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = conv1(x, mask_down, t)
            x = conv2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for conv1, conv2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = conv1(x, mask_up, t)
            x = conv2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(nn.Module):
    def __init__(self, n_mel, channels, mults, beta_min=0.05, beta_max=20):
        super(Diffusion, self).__init__()
        self.n_mel = n_mel
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.estimator = ScoreNet(channels, dim_mults=mults)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t[:, None, None]
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mu):
        mask = torch.ones((z.size(0), 1, z.size(-1)), device=z.device)
        xt = z * mask

        shape = z.shape
        device = z.device
        offset = 1e-5
        solver = 'DOP853'

        def score_eval_wrapper(x, time_steps):
            x = torch.tensor(x, device=device, dtype=torch.float).view(*shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float).view(x.size(0))
            score = self.estimator(x, mask, mu, time_steps)
            return score.cpu().numpy().reshape(-1).astype(np.float64)

        def ode_func(t, x):
            time_steps = np.ones(shape[0]) * t
            time = time_steps[:, None, None]
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            dxt = 0.5 * (mu.cpu().numpy().reshape(-1) - x - score_eval_wrapper(x, time_steps))
            dxt = dxt * noise_t
            return dxt

        t_span = (1., offset)
        res = solve_ivp(ode_func, t_span, xt.cpu().numpy().reshape(-1).astype(np.float64),
                        rtol=1e-5, atol=1e-5, method=solver)
        xt = torch.tensor(res.y[:, -1], dtype=torch.float, device=device).view(*shape)
        return xt

    def loss_t(self, x0, mask, mu, t):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_mel)
        return loss, xt

    def compute_loss(self, x0, mask, mu, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t)
