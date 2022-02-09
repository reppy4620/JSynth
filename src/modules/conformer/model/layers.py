import torch.nn as nn
import torch.nn.functional as F

from ...common.model.layers.common import LayerNorm


class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels * 2, 1)
        self.glu = GLU(dim=1)
        self.depth_wise_conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.layer_norm(x)
        x = self.conv1(x) * x_mask
        x = self.glu(x)
        x = self.depth_wise_conv(x) * x_mask
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv2(x) * x_mask
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, self.dim)


class PostNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, kernel_size=5):
        super(PostNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(hidden_channels, in_channels, kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x, x_mask):
        x = self.layers(x)
        x *= x_mask
        return x
