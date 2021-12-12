from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int, dim_out: int = None):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)
        if dim_out is not None:
            self.out = nn.Linear(dim_v, dim_out)
        else:
            self.out = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.out is not None:
            return self.out(scaled_dot_product_attention(self.q(query), self.k(key), self.v(value)))
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        #self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None) -> Tensor:
        if key == None:
            key = query
        if value == None:
            value = query
        return torch.cat([h(query, key, value) for h in self.heads], dim=-1)
