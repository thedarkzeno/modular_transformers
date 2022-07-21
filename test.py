# # from torch.nn.functional import layer_norm
# from model import Model
# # from layers.transformer import TransformerLayer
# from layers import TransformerLayer, gMLPBlock
# from config import Config, LayerConfig
import numpy as np
import torch
import time
from model import ModelForMaskedLM, Config
# import matplotlib.pyplot as plt
# from mlm_fly import mask_with_prob

config1 = Config(vocab_size=50265,
                layers_attention_type=["lmu"]*6)

config2 = Config(vocab_size=50265,
                layers_attention_type=["attention"]*6)

model1  = ModelForMaskedLM(config1)

model2 = ModelForMaskedLM(config2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model1))
print(count_parameters(model2))

# modelConfig = Config(columns=1, hidden_size=768)
# layerConfig = LayerConfig(hidden_size=768, intermediate_size=3072,
#                           num_attention_heads=12, attention_type="fourier")
# transformerLayers = [TransformerLayer(layerConfig) for _ in range(6)]
# transformerLayers = transformerLayers + [gMLPBlock(layerConfig) for _ in range(6)]


# model = Model(config=modelConfig, layers=transformerLayers)
# model.save_pretrained("./model/")

# [torch.randint(0, 100, (1,512)) for _ in range(5)]
inputs = torch.randint(0, 100, (5, 512))
# print(inputs.shape)

t1 = time.time()
for i in range(10):
    res = model1(inputs)
t2 = time.time()
# print(res.shape)
print(t2-t1)

t1 = time.time()
for i in range(10):
    res = model2(inputs)
t2 = time.time()
# print(res.shape)
print(t2-t1)
