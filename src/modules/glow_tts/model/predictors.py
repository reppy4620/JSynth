import torch
import torch.nn as nn

from .layers import LayerNorm
from ...common.utils import sequence_mask, generate_path


class VarianceAdopter(nn.Module):
    def __init__(self, in_channels, channels, dropout):
        super(VarianceAdopter, self).__init__()
        self.duration_predictor = VariancePredictor(
            in_channels=in_channels,
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x,
        x_mu,
        x_logs,
        x_mask,
        path
    ):
        dur_pred = self.duration_predictor(x.detach(), x_mask)
        z_mu = self.length_regulator(x_mu, path)
        z_logs = self.length_regulator(x_logs, path)
        return z_mu, z_logs, dur_pred

    def infer(self, x, x_mu, x_logs, x_mask):
        dur_pred = self.duration_predictor(x, x_mask)
        dur_pred = torch.round(dur_pred).clamp_min(1) * x_mask
        y_lengths = torch.clamp_min(torch.sum(dur_pred, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.device)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        path = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(1))

        z_mu = self.length_regulator(x_mu, path)
        z_logs = self.length_regulator(x_logs, path)
        return z_mu, z_logs, y_mask


class VariancePredictor(nn.Module):
    def __init__(self, in_channels, channels, n_layers, kernel_size, dropout):
        super(VariancePredictor, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else channels, channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                LayerNorm(channels),
                nn.Dropout(dropout)
            ) for i in range(n_layers)
        ])

        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x)
            x *= x_mask
        x = self.out(x)
        x *= x_mask
        return x


class LengthRegulator(nn.Module):
    def forward(self, x, path):
        x = torch.bmm(x, path)
        return x
