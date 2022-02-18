import torch
import torch.nn as nn

from .predictors import VarianceAdopter
from ...glow_tts.model.layers import PreNet
from ...glow_tts.model.glow import Glow
from ...glow_tts.model.loss import mle_loss, duration_loss
from ...common.model.layers.embedding import EmbeddingLayer, RelPositionalEncoding
from ...common.model.layers.transformer import Transformer
from ...common.utils import sequence_mask, generate_path


class GlowTTSWithF0Model(nn.Module):
    def __init__(self, params):
        super(GlowTTSWithF0Model, self).__init__()

        self.emb = EmbeddingLayer(**params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.pre_net = PreNet(params.encoder.channels)
        self.encoder = Transformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Transformer(**params.encoder)
        self.proj_mu = nn.Conv1d(params.encoder.channels, params.n_mel, 1)
        self.glow = Glow(in_channels=params.n_mel, **params.glow)

    def forward(self, inputs, noise_scale=0.667):
        *labels, x_length = inputs
        x = self.emb(*labels)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x, z_mask = self.variance_adopter.infer(
            x,
            x_mask
        )
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, z_mask)
        z_mu = self.proj_mu(x) * z_mask
        z_logs = torch.zeros_like(z_mu)

        z = (z_mu + torch.exp(z_logs) * torch.randn_like(z_mu) * noise_scale) * z_mask

        y, *_ = self.glow.backward(z, z_mask)
        return y

    def compute_loss(self, batch):
        (
            *labels,
            x_length,
            y,
            y_length,
            duration,
            pitch,
            energy
        ) = batch
        x = self.emb(*labels)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        z_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mask,
            z_mask,
            pitch,
            energy,
            path
        )
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, z_mask)
        z_mu = self.proj_mu(x) * z_mask
        z_logs = torch.zeros_like(z_mu)

        z, log_df_dz, z_mask = self.glow(y, z_mask)

        z_mu = z_mu[:, :, :z.size(-1)]
        z_logs = z_logs[:, :, :z.size(-1)]
        duration = torch.sum(path, dim=-1)

        loss_mle = mle_loss(z, z_mu, z_logs, log_df_dz, z_mask)
        loss_dur = duration_loss(dur_pred, duration, x_length)
        loss_pitch = duration_loss(pitch_pred, pitch, y_length)
        loss_energy = duration_loss(energy_pred, energy, y_length)
        loss = loss_mle + loss_dur + loss_pitch + loss_energy

        return dict(
            loss=loss,
            mle=loss_mle,
            duration=loss_dur,
            pitch=loss_pitch,
            energy=loss_energy
        )

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()
