import math

import torch
from torch import nn
import torch.nn.functional as F

class WeightGenerator(nn.Module):
    def __init__(self, emb_dim, gen_hidden_layer, out_channels, gen_depth, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(gen_depth):
            in_dim = emb_dim if i == 0 else gen_hidden_layer * 2**(i-1)
            out_dim = out_channels if i == gen_depth -1 else gen_hidden_layer * 2**i
            self.layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            if i < gen_depth - 1 :
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        x = x.to(torch.float)
        for layer in self.layers:
            x = layer(x)
        return x
    
class BatchNormEmb(nn.Module):
    def __init__(self, num_features, model_type, device, eps=1e-05, momentum=0.1, **kwargs):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features
        self.model_type = model_type
        self.device = device
        self.kwargs = kwargs

    def init_norm_params(self):
        self.register_buffer('running_mean', torch.zeros(self.num_features, device=self.device))
        self.register_buffer('running_var', torch.ones(self.num_features, device=self.device))
        if self.model_type == 'vanilla':
            self.weight = nn.Parameter(torch.empty(self.num_features, device=self.device))
            self.bias = nn.Parameter(torch.empty(self.num_features, device=self.device))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.weight_generator = WeightGenerator(out_channels=2 * self.num_features, **self.kwargs)

    def update_weights_with_emb(self, emb):
        if emb is not None:
            gen_weights = self.weight_generator(emb)
            self.weight = 1 + gen_weights[:self.num_features]
            self.bias = gen_weights[self.num_features:]

    def forward(self, x):
        self.weight = self.weight.to(x.dtype)
        self.bias = self.bias.to(x.dtype)
        x = F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var, weight=self.weight, bias=self.bias, training=self.training, momentum=self.momentum, eps=self.eps)
        return x
    
class InstanceNormEmb(nn.Module):
    def __init__(self, num_features, model_type, device, eps=1e-05, momentum=0.1, in_bias=True, **kwargs):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features
        self.model_type = model_type
        self.kwargs = kwargs
        self.in_bias = in_bias
        self.device = device

    def init_norm_generator_params(self):
        if self.model_type == 'vanilla':
            self.weight = nn.Parameter(torch.empty(self.num_features, device=self.device))
            self.bias = nn.Parameter(torch.empty(self.num_features, device=self.device))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.weight_generator = WeightGenerator(out_channels=2 * self.num_features, **self.kwargs)

    def update_weights_with_emb(self, emb):
        if emb is not None:
            gen_weights = self.weight_generator(emb)
            self.weight = 1 + gen_weights[:self.num_features]
            self.bias = gen_weights[self.num_features:]

    def forward(self, x):
        self.weight = self.weight.to(x.dtype)
        self.bias = self.bias.to(x.dtype)
        x = F.instance_norm(x, weight=self.weight, bias=self.in_bias, momentum=self.momentum, eps=self.eps)
        return x
    
class EmbNormLayer(nn.Module):
    def __init__(self, num_features, norm_layer, **kwargs):
        super().__init__()
        if norm_layer == 'bn':
            self.norm_layer = BatchNormEmb(num_features=num_features, **kwargs)
        elif norm_layer == 'in':
            self.norm_layer = InstanceNormEmb(num_features=num_features, **kwargs)
    
    def forward(self, x):
        return self.norm_layer(x)