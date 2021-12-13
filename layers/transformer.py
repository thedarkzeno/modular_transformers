import torch
from .attention import MultiHeadAttention, fourier_transform, AttentionHead
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


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "TransformerLayer"
        self.config = config
        # define the type of attention
        if self.config.attention_type == "fourier":
            self.attention=fourier_transform
        elif self.config.attention_type == "selfAttention":
            assert(config.hidden_size % config.num_attention_heads ==
                0), "hidden_size should be divisible by num_attention_heads"
            self.head_size = config.hidden_size // config.num_attention_heads
            self.attention = MultiHeadAttention(
                config.num_attention_heads, config.hidden_size, self.head_size, self.head_size)
        self.attn = AttentionHead(config.hidden_size, config.attn_dim, config.attn_dim, config.hidden_size) if config.tinyAtt else None

        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm_res = nn.LayerNorm(config.hidden_size) if config.tinyAtt else None
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        residual = x
        att_res = self.attn(x, x , x) if self.attn is not None else None
        x = self.attention(x)
        x = self.norm1(x + residual)
        if self.attn is not None:
            x = self.norm_res(x + att_res)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out