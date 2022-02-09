import math
import random
import torch
import torch.nn as nn

from .layers import PreNet
from .predictors import VarianceAdopter
from ...common.model.layers.embedding import EmbeddingLayer, RelPositionalEncoding
from ...common.model.layers.transformer import Transformer
from .diffusion import Diffusion
from ...glow_tts.monotonic_align import maximum_path
from ...common.utils import sequence_mask


class GradTTSModel(nn.Module):
    def __init__(self, params):
        super(GradTTSModel, self).__init__()

        self.n_mel = params.n_mel
        self.segment_length = params.segment_length
        self.adjust_length = 2 ** len(params.decoder.mults)

        self.emb = EmbeddingLayer.from_config(params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.pre_net = PreNet(params.encoder.channels)
        self.encoder = Transformer(**params.encoder)
        self.proj_mu = nn.Conv1d(params.encoder.channels, params.n_mel, 1)
        self.variance_adopter = VarianceAdopter(*params.variance_adopter)
        self.decoder = Diffusion(n_mel=params.n_mel, **params.decoder)

    def compute_loss(self, batch):
        (
            *labels,
            x_length,
            y,
            y_length
        ) = batch
        x = self.emb(*labels)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_mel
            factor = -0.5 * torch.ones(x_mu.shape, dtype=x_mu.dtype, device=x_mu.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * x_mu).transpose(1, 2), y)
            mu_square = torch.sum(factor * (x_mu ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            path = maximum_path(log_prior, attn_mask.squeeze(1)).detach()

        y_mu, dur_pred = self.variance_adopter(x_mu, x_length, x_mask, path.squeeze(1))

        duration = torch.sum(path.unsqueeze(1), dim=-1) * x_mask
        duration_loss = torch.sum((dur_pred - duration) ** 2) / torch.sum(x_length)

        prior_loss = torch.sum(0.5 * ((y - y_mu) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_mel)

        y, y_mu, y_mask = self.rand_slice(y_length, y, y_mu, y_mask)

        diff_loss, _ = self.decoder.compute_loss(y, y_mask, y_mu)
        loss = diff_loss + prior_loss + duration_loss

        return dict(
            loss=loss,
            diffusion=diff_loss,
            prior=prior_loss,
            duration=duration_loss
        )

    def infer(self, inputs, n_time_steps=10, temperature=1.0, use_solver=False):
        *labels, x_length = inputs
        x = self.emb(*labels)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)

        mu, z_mask = self.variance_adopter.infer(x, x_mu, x_mask)

        noise = torch.randn_like(mu) / temperature
        z = mu + noise
        z, mu = self.preprocess(z.size(-1), z, mu)
        y = self.decoder.reverse_diffusion(z, mu, n_time_steps, use_solver)
        return y

    def rand_slice(self, length, *args, seg_size=128):
        min_length = (length - seg_size).clamp_min(0).min()
        b = random.randint(0, min_length)
        e = b + seg_size
        return (x[..., b:e] for x in args)

    def preprocess(self, length, *args):
        new_length = (length // self.adjust_length) * self.adjust_length
        return (x[..., :new_length] for x in args)
