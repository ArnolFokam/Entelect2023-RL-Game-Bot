from typing import Optional
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(
        self, 
        num_actions,
        observation_shape, 
        hidden_dim: Optional[int] = 128,
        num_hidden_layers: Optional[int] = 1) -> None:
        
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(num_hidden_layers, int) and num_hidden_layers >= 1 
        assert len(observation_shape) == 1
        assert num_actions > 1
        
        super().__init__()
        
        # build the neural network of the policy network
        self.layers =  [] 
        
        for i in range(num_hidden_layers):
            input_dim = hidden_dim if i > 0 else observation_shape[0]
            output_dim = hidden_dim if i < num_hidden_layers - 1 else num_actions
            
            if 0 < i < num_hidden_layers - 1:
                self.layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
            else:
                self.layers.append(nn.Linear(input_dim, output_dim))
                
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
        
    

policy_nets = {
    "mlp": MLP,
    "cnn": None
}