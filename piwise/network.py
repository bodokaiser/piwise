import torch
import torch.nn as nn
import torch.nn.init as init
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

class BasicSegNetUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.up(x)


class BasicSegNetDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

    def forward(self, x):
        return self.down(x)


class BasicSegNet(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.down1 = BasicSegNetDown(num_channels, 64)
        self.down2 = BasicSegNetDown(64, 64)
        self.down3 = BasicSegNetDown(64, 64)
        self.down4 = BasicSegNetDown(64, 64)
        self.up4 = BasicSegNetUp(64, 64)
        self.up3 = BasicSegNetUp(128, 64)
        self.up2 = BasicSegNetUp(128, 64)
        self.up1 = BasicSegNetUp(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up4 = self.up4(down4)
        up3 = self.up3(torch.cat([down3, up4], 1))
        up2 = self.up2(torch.cat([down2, up3], 1))
        up1 = self.up1(torch.cat([down1, up2], 1))

        return self.final(up1)