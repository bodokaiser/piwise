import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple(nn.Module):

    def __init__(self, num_channels, num_classes):
        num_features = num_classes*num_channels*12

        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        return self.conv2(x)


class Fcn8(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.conv1a = nn.Conv2d(num_channels, 64, 3, padding=100)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3c = nn.Conv2d(256, 256, 3, padding=1)
        self.score1 = nn.Conv2d(256, num_classes, 1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4c = nn.Conv2d(512, 512, 3, padding=1)
        self.score2 = nn.Conv2d(512, num_classes, 1)

        self.fcon1a = nn.Conv2d(512, 4096, 7)
        self.fcon1b = nn.Conv2d(4096, 4096, 1)
        self.score3a = nn.Conv2d(4096, num_classes, 1)
        self.score3b = nn.ConvTranspose2d(num_classes, num_classes, 4,
            stride=2, bias=False)

    def forward(self, x):
        conv1 = F.relu(self.conv1a(x), inplace=True)
        conv1 = F.relu(self.conv1b(conv1), inplace=True)
        conv1 = F.max_pool2d(conv1, 2, 2)

        conv2 = F.relu(self.conv2a(conv1), inplace=True)
        conv2 = F.relu(self.conv2b(conv2), inplace=True)
        conv2 = F.max_pool2d(conv2, 2, 2)

        conv3 = F.relu(self.conv3a(conv2), inplace=True)
        conv3 = F.relu(self.conv3b(conv3), inplace=True)
        conv3 = F.relu(self.conv3c(conv3), inplace=True)
        conv3 = F.max_pool2d(conv3, 2, 2)

        conv4 = F.relu(self.conv4a(conv3), inplace=True)
        conv4 = F.relu(self.conv4b(conv4), inplace=True)
        conv4 = F.relu(self.conv4c(conv4), inplace=True)
        conv4 = F.max_pool2d(conv4, 2, 2)

        fcon1 = F.relu(self.fcon1a(conv4), inplace=True)
        fcon1 = F.dropout(fcon1, .5, inplace=True)
        fcon1 = F.relu(self.fcon1b(fcon1), inplace=True)
        fcon1 = F.dropout(fcon1, .5, inplace=True)