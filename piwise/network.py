import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        num_features = num_classes*num_channels*12
        self.conv1 = nn.Conv2d(num_channels, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        return self.conv2(x)