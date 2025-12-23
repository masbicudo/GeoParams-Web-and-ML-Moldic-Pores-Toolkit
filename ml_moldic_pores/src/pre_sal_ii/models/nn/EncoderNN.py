import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderNN(nn.Module):
    def __init__(self, initial_dim=3*32*32):
        super().__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(initial_dim, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 8*8),
            nn.ReLU(),
            nn.Linear(8*8, 4*4),
            nn.ReLU(),
            nn.Linear(4*4, 2*2),
            nn.ReLU(),
            nn.Linear(2*2, 1*1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x
