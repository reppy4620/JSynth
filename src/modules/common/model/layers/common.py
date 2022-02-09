import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, channels, dropout):
        super(FFN, self).__init__()

        self.norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels * 4, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels * 4, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        x = self.dropout(x)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x