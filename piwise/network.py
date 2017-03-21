import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 1, 3, 1, 1)

    def forward(self, image):
        return self.conv(image)

class CrossEntropySoftmax2d(nn.Module):

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        return F.cross_entropy(inputs, targets)
