import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

class FCN8(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())

        self.feats = nn.Sequential(*feat[0:9])
        self.feat3 = nn.Sequential(*feat[10:16])
        self.feat4 = nn.Sequential(*feat[17:23])
        self.feat5 = nn.Sequential(*feat[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        score = F.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        return F.upsample_bilinear(score, x.size()[2:])


class FCN16(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])


class FCN32(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)

        return F.upsample_bilinear(score, x.size()[2:])


class UNetUp(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.down1 = UNetDown(3, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.up4 = UNetUp(1024, 512, 256)
        self.up3 = UNetUp(512, 256, 128)
        self.up2 = UNetUp(256, 128, 64)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(torch.cat([
            center, F.upsample_bilinear(down4, center.size()[2:])], 1))
        up3 = self.up3(torch.cat([
            up4, F.upsample_bilinear(down3, up4.size()[2:])], 1))
        up2 = self.up2(torch.cat([
            up3, F.upsample_bilinear(down2, up3.size()[2:])], 1))
        up1 = self.up1(torch.cat([
            up2, F.upsample_bilinear(down1, up2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(up1), x.size()[2:])


class SegNet1Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.up(x)


class SegNet1Down(nn.Module):

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


class SegNet1(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.down1 = SegNet1Down(3, 64)
        self.down2 = SegNet1Down(64, 64)
        self.down3 = SegNet1Down(64, 64)
        self.down4 = SegNet1Down(64, 64)
        self.up4 = SegNet1Up(64, 64)
        self.up3 = SegNet1Up(128, 64)
        self.up2 = SegNet1Up(128, 64)
        self.up1 = SegNet1Up(128, 64)
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


class SegNet2Up(nn.Module):

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


class SegNet2Down(nn.Module):

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


class SegNet2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.down1 = SegNet2Down(3, 64, layers=1)
        self.down2 = SegNet2Down(64, 128, layers=1)
        self.down3 = SegNet2Down(128, 256, layers=2)
        self.down4 = SegNet2Down(256, 512, layers=2)
        self.down5 = SegNet2Down(512, 512, layers=2)
        self.up5 = SegNet2Up(512, 512, layers=1)
        self.up4 = SegNet2Up(1024, 256, layers=1)
        self.up3 = SegNet2Up(512, 128, layers=1)
        self.up2 = SegNet2Up(256, 64, layers=0)
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


class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=60):
        super().__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(upsize)
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        '''

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False

        self.layer5a = PSPDec(2048, 512, 60)
        self.layer5b = PSPDec(2048, 512, 30)
        self.layer5c = PSPDec(2048, 512, 20)
        self.layer5d = PSPDec(2048, 512, 10)

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        print('x', x.size())
        x = self.conv1(x)
        print('conv1', x.size())
        x = self.layer1(x)
        print('layer1', x.size())
        x = self.layer2(x)
        print('layer2', x.size())
        x = self.layer3(x)
        print('layer3', x.size())
        x = self.layer4(x)
        print('layer4', x.size())
        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))
        print('final', x.size())

        return F.upsample_bilinear(final, x.size()[2:])