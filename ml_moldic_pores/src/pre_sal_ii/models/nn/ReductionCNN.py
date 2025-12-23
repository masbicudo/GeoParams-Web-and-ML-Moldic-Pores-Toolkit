import torch
import torch.nn as nn
import torch.nn.functional as F

class ReductionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 13 * 13, 512)  # Flattened from 128 feature maps of size 13x13
        self.fc2 = nn.Linear(512, 1)              # Single output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # Conv1 + ReLU + MaxPool # 101x101 -> 51x51
        x = self.pool(F.relu(self.conv2(x)))     # Conv2 + ReLU + MaxPool # 51x51 -> 26x26
        x = self.pool2(F.relu(self.conv3(x)))    # Conv3 + ReLU + MaxPool # 26x26 -> 13x13
        x = torch.flatten(x, start_dim=1)        # Flatten for the fully connected layers
        x = F.relu(self.fc1(x))                  # FC1 + ReLU
        x = torch.sigmoid(self.fc2(x))           # Sigmoid activation for probability output
        return x
