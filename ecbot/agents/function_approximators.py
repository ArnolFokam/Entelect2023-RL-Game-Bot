import numpy as np
from typing import Optional

import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNN(nn.Module):
    def __init__(
        self,
        output_shape,
        *args, **kwargs):
        
        assert len(output_shape) == 1
        
        super().__init__()  
        
        # TODO: Build the CNN
        self.cnn = None
        
        # TODO: Build the fully connected layers
        self.mlp = None
        
    def forward(self, x):
        x = self.cnn(x)
        return self.mlp(x.view(x.size(0), -1))

class MLP(nn.Module):
    
    def __init__(
        self, 
        input_shape,
        output_shape, 
        hidden_dim: Optional[int] = 128,
        num_hidden_layers: Optional[int] = 1,
        *args, **kwargs) -> None:
        
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(num_hidden_layers, int) and num_hidden_layers >= 1 
        assert len(input_shape) == 1
        assert len(output_shape) == 1
        
        super().__init__()
        
        # build the neural network of the policy network
        self.layers =  [] 
        
        for i in range(num_hidden_layers):
            input_dim = hidden_dim if i > 0 else input_shape[0]
            output_dim = hidden_dim if i < num_hidden_layers - 1 else output_shape[0]
            
            if 0 < i < num_hidden_layers - 1:
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