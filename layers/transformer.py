import torch.nn as nn
from .selfAttention import MultiHeadAttention


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "TransformerLayer"
        self.config = config
        assert(config.hidden_size % config.num_attention_heads ==
               0), "hidden_size should be divisible by num_attention_heads"
        self.head_size = config.hidden_size // config.num_attention_heads
        self.MultiHeadAttention = MultiHeadAttention(
            config.num_attention_heads, config.hidden_size, self.head_size, self.head_size)
        # self output
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        # intermediate
        self.intermediate = nn.Linear(
            config.hidden_size, config.intermediate_size)
        # output
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.outputLayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.MultiHeadAttention(x, x, x)
        x = self.dense(x)
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.intermediate(x)
        x = self.output(x)
        x = self.outputLayerNorm(x)
        x = self.dropout2(x)
        return x
