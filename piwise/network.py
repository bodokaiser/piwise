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


class SegNetUp(nn.Module):

    def __init__(self, in_channels, out_channels, layers):
        super().__init__()

        up = [
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        up += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * layers
        up += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.up = nn.Sequential(*up)

    def forward(self, x):
        return self.up(x)


class SegNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, layers):
        super().__init__()

        down = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        down += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ] * layers
        down += [
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        ]
        self.down = nn.Sequential(*down)

    def forward(self, x):
        return self.down(x)


class SegNet(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.down1 = SegNetDown(num_channels, 64, layers=1)
        self.down2 = SegNetDown(64, 128, layers=1)
        self.down3 = SegNetDown(128, 256, layers=2)
        self.down4 = SegNetDown(256, 512, layers=2)
        self.down5 = SegNetDown(512, 512, layers=2)
        self.up5 = SegNetUp(512, 512, layers=1)
        self.up4 = SegNetUp(1024, 256, layers=1)
        self.up3 = SegNetUp(512, 128, layers=1)
        self.up2 = SegNetUp(256, 64, layers=0)
        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        up5 = self.up5(down5)
        up4 = self.up4(torch.cat([down4, up5], 1))
        up3 = self.up3(torch.cat([down3, up4], 1))
        up2 = self.up2(torch.cat([down2, up3], 1))
        up1 = self.up1(torch.cat([down1, up2], 1))

        return self.final(up1)