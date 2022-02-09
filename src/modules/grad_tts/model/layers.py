import torch.nn as nn

from ...common.model.layers.common import LayerNorm


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.5):
        super(ConvLayer, self).__init__()
        self.norm = LayerNorm(out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv(x * x_mask)
        x = self.act(x)
        x = self.dropout(x)
        return x


class PreNet(nn.Module):
    def __init__(self, channels, n_layers=3, kernel_size=5, dropout=0.5):
        super(PreNet, self).__init__()

        self.layers = nn.ModuleList([
            ConvLayer(
                channels,
                channels,
                kernel_size,
                dropout
            ) for _ in range(n_layers)
        ])
        self.out = nn.Conv1d(channels, channels, 1)
        self.out.weight.data.zero_()
        self.out.bias.data.zero_()

    def forward(self, x, x_mask):
        residual = x
        for layer in self.layers:
            x = layer(x, x_mask)
        x = residual + self.out(x)
        x *= x_mask
        return x
