import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, config.hidden_size)
        
    def forward(self, Input, mask=None):
        seq_length = Input.size()[1]
        batch_size = Input.size()[0]
        position_ids = self.position_ids.expand((batch_size, -1))
        position_ids = position_ids[:, 0 : seq_length + 0]
        x = self.embedding(Input)
        x += self.position_embeddings(position_ids)
        if mask is not None:
            x += self.mask_embedding(mask)
        return x