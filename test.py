from torch.nn.functional import layer_norm
from model import Model
from layers.transformer import TransformerLayer
from layers.fnet import FNetLayer
from config import Config, LayerConfig
import numpy as np
import torch
import time
import matplotlib.pyplot as plt


modelConfig = Config()
layerConfig = LayerConfig()

# model = Model.from_pretrained("./model")
# model.save_pretrained("./model2")

transformerLayers = [TransformerLayer(layerConfig) for _ in range(2)]
fnetLayers = [FNetLayer(layerConfig) for _ in range(10)]

transformerLayers.extend(fnetLayers)

model = Model(config=modelConfig, layers=transformerLayers)

# model.save_pretrained("./model")
# size = []
# times = []
# times2 = []
# times_lin = []
# lt =0
# initial_value=100

inputs = torch.randint(0, 100, (5,100))#[torch.randint(0, 100, (1,512)) for _ in range(5)]
# inputs = [torch.randint(0, 100, (10,10)) for _ in range(5)]
# inputs = torch.cat(inputs).view(5, -1)
# print('input ',inputs.shape)

# print(torch.cat(inputs).view(10, len(inputs), 1, -1))
# print(torch.cat(inputs).view(10, len(inputs), 1, -1).shape)

res = model(inputs)
print(res.shape)
# print(torch.zeros_like(torch.tensor([1,2,3,4,5,6,7,8,9,10])).float().uniform_(0, 1) < 0.15)

# for s in range(initial_value, 500, 100):
    

#     x = torch.randint(0, 100, (10,s))

#     # print(x)
#     t1 = time.time()
#     for i in range(20):
#         model(x)
#     t2 = time.time()-t1
    
#     if lt==0:
#         lt=t2
#     size.append(s)
#     times.append(t2)
#     times2.append(lt*(s/initial_value)**2)
#     times_lin.append(lt*(s/initial_value))
#     print(s, t2, lt*(s/initial_value)**2, lt*(s/initial_value))
# plt.plot(size, times)
# plt.plot(size, times2)
# plt.plot(size, times_lin)
# plt.show()
# # x = torch.randint(0, 100, (1,1000))

# # # print(x)
# # t1 = time.time()
# # for i in range(10):
# #     model(x)
# # print("time: ", time.time()-t1)

# print(model(x).shape)