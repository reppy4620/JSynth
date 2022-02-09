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


class WaveNet(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dilation_rate=1, gin_channels=0, dropout=0):
        super(WaveNet, self).__init__()

        self.channels = channels
        self.num_layers = num_layers

        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            conv = nn.Conv1d(channels, channels * 2, kernel_size, padding=padding, dilation=dilation)
            conv = nn.utils.weight_norm(conv)
            self.dilated_convs.append(conv)

        self.out_convs = nn.ModuleList()
        for i in range(num_layers):
            conv = nn.Conv1d(channels, channels * 2 if i < num_layers-1 else channels, 1)
            conv = nn.utils.weight_norm(conv)
            self.out_convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        if g is not None:
            g = self.cond_layer(g)
        out = 0
        for i, (d_conv, o_conv) in enumerate(zip(self.dilated_convs, self.out_convs)):
            x_in = d_conv(x)
            if g is not None:
                x_in += g
            x_in_a, x_in_b = x_in.chunk(2, dim=1)
            x_in = x_in_a.sigmoid() * x_in_b.tanh()
            if i < self.num_layers - 1:
                o1, o2 = o_conv(x_in).chunk(2, dim=1)
                x = (x + o1) * x_mask
                x = self.dropout(x)
                out += o2 * x_mask
            else:
                out += o_conv(x_in)
        return out * x_mask

    def remove_weight_norm(self):
        for l in self.dilated_convs:
            nn.utils.remove_weight_norm(l)
        for l in self.out_convs:
            nn.utils.remove_weight_norm(l)
