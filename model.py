import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim) / (input_dim ** 0.5)) # Xavier
        self.b = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x, return_h=False):
        h = x @ self.W
        out = h @ self.W.T
        out = F.relu(out + self.b)
        if return_h:
            return out, h
        return out
