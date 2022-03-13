# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.integrate import solve_ivp
from einops import rearrange


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


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


class Rezero(nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            Mish()
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(
            Mish(),
            torch.nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
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
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GaussianFourierProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim // 2), requires_grad=False)

    def forward(self, x, scale=16.):
        x_proj = scale * x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, pos_emb_type='sinusoid'):
        super(ScoreNet, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups

        if pos_emb_type == 'sinusoid':
            self.pe_scale = 1000.
            self.time_pos_emb = SinusoidalPosEmb(dim)
        elif pos_emb_type == 'gaussian':
            self.pe_scale = 16.
            self.time_pos_emb = GaussianFourierProjection(dim)
        else:
            raise ValueError('pos_emb_type must be sinusoid or gaussian.')
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                    ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else torch.nn.Identity()
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_in))),
                    Upsample(dim_in)
                ])
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t):

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        x = torch.stack([mu, x], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
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
    def __init__(self, n_mel, channels, mults, beta_min=0.05, beta_max=20, pos_emb_type='sinusoid'):
        super(Diffusion, self).__init__()
        self.n_mel = n_mel
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.estimator = ScoreNet(channels, dim_mults=mults, pos_emb_type=pos_emb_type)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mu, n_timesteps, use_solver=False):
        mask = torch.ones((z.size(0), 1, z.size(-1)), device=z.device)
        h = 1.0 / n_timesteps
        xt = z * mask

        if use_solver:
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
                dxt = dxt * noise_t * h
                return dxt

            t_span = (1. - offset, offset)
            res = solve_ivp(ode_func, t_span, xt.reshape(-1).cpu().numpy(), rtol=1e-5, atol=1e-5, method=solver)
            xt = torch.tensor(res.y[:, -1], dtype=torch.float, device=device).view(*shape)
        else:
            for i in range(n_timesteps):
                t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
                time = t.unsqueeze(-1).unsqueeze(-1)
                noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t))
                dxt = dxt * noise_t * h
                xt = (xt - dxt) * mask
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
