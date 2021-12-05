import torch
import torch.fft
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "FNetLayer"
        self.config = config
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out