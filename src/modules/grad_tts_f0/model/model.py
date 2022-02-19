import math
import random
import torch
import torch.nn as nn

from .predictors import VarianceAdopter
from ...glow_tts.model.loss import duration_loss
from ...grad_tts.model.layers import PreNet
from ...grad_tts.model.diffusion import Diffusion
from ...common.model.layers.embedding import EmbeddingLayer, RelPositionalEncoding
from ...conformer.model.conformer import Conformer
from ...common.utils import sequence_mask, generate_path


class GradTTSWithF0Model(nn.Module):
    def __init__(self, params):
        super(GradTTSWithF0Model, self).__init__()

        self.n_mel = params.n_mel
        self.segment_length = params.segment_length
        self.adjust_length = 2 ** len(params.diffusion.mults)

        self.emb = EmbeddingLayer(**params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.pre_net = PreNet(params.encoder.channels)
        self.encoder = Conformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Conformer(**params.encoder)
        self.proj_mu = nn.Conv1d(params.encoder.channels, params.n_mel, 1)
        self.diffusion = Diffusion(n_mel=params.n_mel, **params.diffusion)

    def forward(self, inputs, n_time_steps=100, temperature=1.5, use_solver=False):
        *labels, x_length = inputs
        x = self.emb(*labels)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x, y_mask = self.variance_adopter.infer(
            x,
            x_mask
        )
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        mu = self.proj_mu(x) * y_mask

        noise = torch.randn_like(mu) / temperature
        z = mu + noise
        z, mu = self.preprocess(z.size(-1), z, mu)
        y = self.diffusion.reverse_diffusion(z, mu, n_time_steps, use_solver)
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
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mask,
            y_mask,
            pitch,
            energy,
            path
        )
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        y_mu = self.proj_mu(x) * y_mask

        prior_loss = torch.sum(0.5 * ((y - y_mu) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_mel)

        y, y_mu, y_mask = self.rand_slice(y_length, y, y_mu, y_mask)

        diff_loss, _ = self.diffusion.compute_loss(y, y_mask, y_mu)

        loss_dur = duration_loss(dur_pred, duration, x_length)
        loss_pitch = duration_loss(pitch_pred, pitch, y_length)
        loss_energy = duration_loss(energy_pred, energy, y_length)
        loss = diff_loss + prior_loss + loss_dur + loss_pitch + loss_energy

        return dict(
            loss=loss,
            diffusion=diff_loss,
            prior=prior_loss,
            duration=loss_dur,
            pitch=loss_pitch,
            energy=loss_energy
        )

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()

    def rand_slice(self, length, *args, seg_size=128):
        min_length = (length - seg_size).clamp_min(0).min()
        b = random.randint(0, min_length)
        e = b + seg_size
        return (x[..., b:e] for x in args)

    def preprocess(self, length, *args):
        new_length = (length // self.adjust_length) * self.adjust_length
        return (x[..., :new_length] for x in args)
