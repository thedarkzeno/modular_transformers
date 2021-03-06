from torch import nn

class SpatialGatingUnit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.intermediate_size)
        self.spatial_proj = nn.Conv1d(config.max_position_embeddings, config.max_position_embeddings, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, gate_res = None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        if gate_res is not None:
            v = v + gate_res
        out = u * v
        return (out,)
