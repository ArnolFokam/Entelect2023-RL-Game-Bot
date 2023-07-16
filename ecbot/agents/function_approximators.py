import numpy as np
from typing import Optional

import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNN(nn.Module):
    def __init__(self, arch_cfg, input_shape, output_shape, *args, **kwargs):
        
        assert isinstance(arch_cfg.filters, list) and len(arch_cfg.filters) > 0
        assert len(input_shape) == 3
        assert len(output_shape) == 1
        
        super().__init__()  
        
        # Build the CNN
        cnn_layers = [*self._get_conv_layer(input_shape[0], arch_cfg.filters[0])]
        
        for i in range(len(arch_cfg.filters) - 1):
            cnn_layers.extend(self._get_conv_layer(arch_cfg.filters[i], arch_cfg.filters[i+1]))
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Build the fully connected layers
        self.mlp = layer_init(nn.Linear(arch_cfg.filters[-1] * 4 * 4, output_shape[0]), std=0.01)
        
    def _get_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        
    def forward(self, x):
        x = self.cnn(x)
        return self.mlp(x.view(x.size(0), -1))


class MLP(nn.Module):
    
    def __init__(self, arch_cfg, input_shape, output_shape,*args, **kwargs) -> None:
        
        assert isinstance(arch_cfg.hidden_dim, int) and arch_cfg.hidden_dim > 0
        assert isinstance(arch_cfg.num_hidden_layers, int) and arch_cfg.num_hidden_layers >= 1 
        assert len(input_shape) == 1
        assert len(output_shape) == 1
        
        super().__init__()
        
        # build the neural network of the policy network
        self.layers =  [] 
        
        for i in range(arch_cfg.num_hidden_layers):
            input_dim = arch_cfg.hidden_dim if i > 0 else input_shape[0]
            output_dim = arch_cfg.hidden_dim if i < arch_cfg.num_hidden_layers - 1 else output_shape[0]
            
            if 0 < i < arch_cfg.num_hidden_layers - 1:
                self.layers.extend([layer_init(nn.Linear(input_dim, output_dim)), nn.Tanh()])
            else:
                self.layers.append(layer_init(nn.Linear(input_dim, output_dim), std=0.01))
                
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
        
    

function_approximators = {
    "mlp": MLP,
    "cnn": CNN
}