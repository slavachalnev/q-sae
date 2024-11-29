import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        W = torch.empty(input_dim, hidden_dim)
        # nn.init.kaiming_uniform_(W, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(W)

        W /= W.norm(dim=1, keepdim=True) # Normalise rows

        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x, return_h=False):
        h = x @ self.W
        out = h @ self.W.T
        out = F.relu(out + self.b)
        if return_h:
            return out, h
        return out


if __name__ == "__main__":
    model = Model(1024, 32)
    print(model.W.shape)
    print(model.W.norm(dim=1))