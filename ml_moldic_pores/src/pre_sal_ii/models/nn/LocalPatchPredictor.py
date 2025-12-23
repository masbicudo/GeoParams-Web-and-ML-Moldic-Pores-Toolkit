import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalPatchPredictor(nn.Module):
    def __init__(self, in_channels=30, hidden_dim=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 11→5
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Decoder (upsampling)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.dec2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        
        # Optional normalization
        self.norm = nn.BatchNorm2d(hidden_dim)
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x2p = self.pool(x2)
        
        # Bottleneck
        b = F.relu(self.bottleneck(x2p))
        
        # Decoder
        up = self.up(b)
        # Cropping to match input size (due to odd dimensions like 11x11)
        diffY = x.size(2) - up.size(2)
        diffX = x.size(3) - up.size(3)
        up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        d1 = F.relu(self.dec1(up))
        out = torch.sigmoid(self.dec2(d1))  # grayscale output 0–1
        
        return out
