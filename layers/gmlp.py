from torch import nn, einsum
from torch.nn import functional as F
import torch


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, gate_res = None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        if gate_res is not None:
            v = v + gate_res
        out = u * v
        return out

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class gMLPBlock(nn.Module):
    # def __init__(self, d_model, d_ffn, seq_len):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.channel_proj1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.channel_proj2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.sgu = SpatialGatingUnit(config.intermediate_size, config.max_position_embeddings)
        self.attn = Attention(config.hidden_size, config.intermediate_size, config.attn_dim, config.causal) if config.tinyAtt else None

    def forward(self, x):
        residual = x
        gate_res = self.attn(x) if self.attn is not None else None
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x, gate_res=gate_res)
        x = self.channel_proj2(x)
        out = x + residual
        return out