import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

    def forward(self, x):
        return self.conv(x)


class FCN8(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv3 = FCNConv(128, 256)
        self.conv4 = FCNConv(256, 512)
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
        )

        self.score1 = nn.Conv2d(256, num_classes, 1)
        self.score2 = nn.Conv2d(512, num_classes, 1)
        self.score3 = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        fconn = self.fconn(conv4)

        score1 = self.score1(conv3)
        score2 = self.score2(conv4)
        score3 = self.score3(fconn)

        score = F.upsample_bilinear(score3, score2.size()[2:]) + score2
        score = F.upsample_bilinear(score, score1.size()[2:]) + score1

        return F.upsample_bilinear(score, x.size()[2:])