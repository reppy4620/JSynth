import math
import torch
import torch.nn as nn

from .layers import PreNet
from .predictors import VarianceAdopter
from .glow import Glow
from .loss import mle_loss, duration_loss
from ..monotonic_align import maximum_path
from ...common.model.layers.embedding import EmbeddingLayer, RelPositionalEncoding
from ...common.model.layers.transformer import Transformer
from ...common.utils import sequence_mask


class GlowTTSModel(nn.Module):
    def __init__(self, params):
        super(GlowTTSModel, self).__init__()

        self.emb = EmbeddingLayer(**params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.pre_net = PreNet(params.encoder.channels)
        self.encoder = Transformer(**params.encoder)
        self.proj_mu = nn.Conv1d(params.encoder.channels, params.n_mel, 1)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Glow(in_channels=params.n_mel, **params.decoder)

    def forward(self, inputs, noise_scale=0.667):
        *labels, x_length = inputs
        x = self.emb(*labels)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x) * x_mask
        x_logs = torch.zeros_like(x_mu)

        z_mu, z_logs, z_mask = self.variance_adopter.infer(x, x_mu, x_logs, x_mask)

        z = (z_mu + torch.exp(z_logs) * torch.randn_like(z_mu) * noise_scale) * z_mask

        y, *_ = self.decoder.backward(z, z_mask)
        return y

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
        z_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x) * x_mask
        x_logs = torch.zeros_like(x_mu)

        z, log_df_dz, z_mask = self.decoder(y, z_mask)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))
            logp3 = torch.matmul((x_mu * x_s_sq_r).transpose(1, 2), z)
            logp4 = torch.sum(-0.5 * (x_mu ** 2) * x_s_sq_r, [1]).unsqueeze(-1)
            logp = logp1 + logp2 + logp3 + logp4
            path = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        z_mu, z_logs, dur_pred = self.variance_adopter(x, x_mu, x_logs, x_mask, path.squeeze(1))

        z_mu = z_mu[:, :, :z.size(-1)]
        z_logs = z_logs[:, :, :z.size(-1)]
        duration = torch.sum(path, dim=-1)

        loss_mle = mle_loss(z, z_mu, z_logs, log_df_dz, z_mask)
        loss_dur = duration_loss(dur_pred, duration, x_length)
        loss = loss_mle + loss_dur

        return dict(
            loss=loss,
            mle=loss_mle,
            duration=loss_dur
        )

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()
