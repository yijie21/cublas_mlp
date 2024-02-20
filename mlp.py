import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, hidden_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims, bias=True),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims, bias=True),
                nn.ReLU(inplace=True)
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dims, out_dims, bias=True)
        )
        
        # randomly init the weights and bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.bias, mean=0, std=1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)
    

class MLP_OneLayer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.net = nn.Linear(in_dims, out_dims, bias=True)
        
        # randomly init the weights and bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=1)
                # nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)