import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.model import ModelBase
from ...common.model.layers import EmbeddingLayer, RelPositionalEncoding
from ...common.model.layers.transformer import Transformer
from ...common.utils import sequence_mask, generate_path
from .layers import PostNet
from .conformer import Conformer
from .predictors import VarianceAdopter


class ConformerModel(ModelBase):

    def __init__(self, params):
        super().__init__()

        self.emb = EmbeddingLayer(**params.embedding)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.encoder = Transformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Conformer(**params.decoder)

        self.out_conv = nn.Conv1d(params.decoder.channels, params.n_mel, 1)
        self.post_net = PostNet(params.n_mel)

    def forward(self, inputs):
        *labels, x_length = inputs
        x = self.emb(*labels)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)

        print(x.size(), pos_emb.size(), x_mask.size())
        x = self.encoder(x, pos_emb, x_mask)

        x, y_mask = self.variance_adopter.infer(x, x_mask)
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask
        x_post = self.post_net(x, y_mask)
        x = x + x_post
        return x

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

        x = self.encoder(x, pos_emb, x_mask)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

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
        x = self.out_conv(x)
        x *= y_mask
        x_post = self.post_net(x, y_mask)
        x = x + x_post

        recon_loss = F.mse_loss(x, y)
        recon_post_loss = F.mse_loss(x_post, y)
        duration_loss = F.mse_loss(dur_pred, duration)
        pitch_loss = F.mse_loss(pitch_pred, pitch)
        energy_loss = F.mse_loss(energy_pred, energy)
        loss = recon_loss + recon_post_loss + duration_loss + pitch_loss + energy_loss

        return dict(
            loss=loss,
            recon=recon_loss,
            recon_post=recon_post_loss,
            duration=duration_loss,
            pitch=pitch_loss,
            energy=energy_loss
        )
