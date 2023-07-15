from typing import Optional
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self, output_shape, *args, **kwargs) -> None:
        super().__init__()
        if output_shape[0] == 1:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3520, 128),
                nn.Tanh(),
                nn.Linear(128, output_shape[0])
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3520, 512),
                nn.Tanh(),
                nn.Linear(512, output_shape[0]),
                nn.Softmax()
            )
    def forward(self, x):
        return self.network(x)

class MLP(nn.Module):
    
    def __init__(
        self, 
        input_shape,
        output_shape, 
        hidden_dim: Optional[int] = 128,
        num_hidden_layers: Optional[int] = 1) -> None:
        
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
                self.layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
            else:
                self.layers.append(nn.Linear(input_dim, output_dim))
                
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = x.squeeze()
        return self.layers(x)
        
    

function_approximators = {
    "mlp": MLP,
    "cnn": CNN,
}