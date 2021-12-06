import torch
import torch.nn as nn
from pathlib import Path
from .config import Config, LayerConfig
from .layers import *
import json

class Model(nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList(layers)
        # self.output = nn.Linear(config.hidden_size * config.columns, config.vocab_size)
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    
    def save_pretrained(self, path):
        Path(path+"/").mkdir(parents=True, exist_ok=True)
        with open(path+'/config.json', 'w') as outfile:
            data = self.config.getDict()
            data["model"] = "Model"
            data["layers"] = []
            for layer in self.layers:
                layer_dict = layer.config.getDict()
                layer_dict["name"] = layer.name
                data["layers"].append(layer_dict)
            json.dump(data, outfile)
        torch.save(self.state_dict(), path+"/model.bin")
    
    def from_pretrained(path):
        with open(path+'/config.json') as json_file:
            data = json.load(json_file)
        base_config = Config()
        base_config.fromDict(data)
        layers = []
        for layer in data["layers"]:
            config = LayerConfig()
            config.fromDict(layer)
            layer_to_add = transformer.TransformerLayer(config)
            layers.append(layer_to_add)
        model = Model(base_config, layers)
        model.load_state_dict(torch.load(path+"/model.bin"))
        return model
        


    def forward(self, Input, mask=None):
        seq_length = Input.size()[1]
        batch_size = Input.size()[0]
        position_ids = self.position_ids.expand((batch_size, -1))
        position_ids = position_ids[:, 0 : seq_length + 0]
        x = self.embedding(Input)
        x += self.position_embeddings(position_ids)
        if mask is not None:
            x += self.mask_embedding(mask)
        out = []
        layers = len(self.layers) / self.config.columns
        for i, layer in enumerate(self.layers):
            if i % layers == 0:
                if i >0:
                    out.append(o)
                o = layer(x)
            else:
                o = layer(o)
        out.append(o)
        y = torch.cat(out, dim=-1)

        # x = self.output(y)
        return y

class ModelForMaskedLM(nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        self.layers=layers
        self.baseModel = Model(config, layers)
        self.output = nn.Linear(config.hidden_size * config.columns, config.vocab_size)
        
    def save_pretrained(self, path):
        Path(path+"/").mkdir(parents=True, exist_ok=True)
        with open(path+'/config.json', 'w') as outfile:
            data = self.config.getDict()
            data["model"] = "ModelForMaskedLM"
            data["layers"] = []
            for layer in self.layers:
                layer_dict = layer.config.getDict()
                layer_dict["name"] = layer.name
                data["layers"].append(layer_dict)
            json.dump(data, outfile)
        torch.save(self.state_dict(), path+"/model.bin")
    
    def from_pretrained(path):
        with open(path+'/config.json') as json_file:
            data = json.load(json_file)
        base_config = Config()
        base_config.fromDict(data)
        layers = []
        for layer in data["layers"]:
            config = LayerConfig()
            config.fromDict(layer)
            layer_to_add = transformer.TransformerLayer(config)
            layers.append(layer_to_add)
        model = ModelForMaskedLM(base_config, layers)
        
        if data["model"] == "Model":
            print("Loading from different Model, some weights might be missing")
            model.baseModel = Model.from_pretrained(path)
        elif data["model"] == "ModelForMaskedLM":
            model.load_state_dict(torch.load(path+"/model.bin"))
        return model

    def forward(self, Input, mask=None):
        x = self.baseModel(Input, mask)
        x = self.output(x)
        return x
