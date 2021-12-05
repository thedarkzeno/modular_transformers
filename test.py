# from torch.nn.functional import layer_norm
from model import Model
# from layers.transformer import TransformerLayer
from layers import TransformerLayer, gMLPBlock
from config import Config, LayerConfig
import numpy as np
import torch
import time
# import matplotlib.pyplot as plt
# from mlm_fly import mask_with_prob


modelConfig = Config(columns=1, hidden_size=768)
layerConfig = LayerConfig(hidden_size=768, intermediate_size=3072,
                          num_attention_heads=12, attention_type="fourier")
transformerLayers = [TransformerLayer(layerConfig) for _ in range(6)]
transformerLayers = transformerLayers + [gMLPBlock(layerConfig) for _ in range(6)]


model = Model(config=modelConfig, layers=transformerLayers)


# [torch.randint(0, 100, (1,512)) for _ in range(5)]
inputs = torch.randint(0, 100, (5, 512))
print(inputs.shape)

t1 = time.time()
for i in range(10):
    res = model(inputs)
t2 = time.time()
print(res.shape)
print(t2-t1)
