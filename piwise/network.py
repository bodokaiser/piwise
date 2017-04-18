import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

def vgg_normalize(images):
    images[:, 0] -= .485
    images[:, 0] /= .229
    images[:, 1] -= .456
    images[:, 1] /= .224
    images[:, 2] -= .406
    images[:, 2] /= .225

class FCN(nn.Module):

    MEAN = [.485, .456, .406]
    STD = [.229, .224, .225]

    def __init__(self, num_classes):
        super().__init__()

        feat = list(models.vgg16(pretrained=True).features.children())

        self.feat1 = nn.Sequential(*feat[0:4])
        self.feat2 = nn.Sequential(*feat[5:9])
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
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def normalize(self, x):
        for i in range(3):
            x[:, i] = (x[:, i] - self.MEAN[i]) / self.STD[i]

    def forward(self, x):
        self.normalize(x)
        x = self.feat1(x)
        x = self.feat2(x)
        x = self.feat3(x)

        return x


class FCN8(FCN):

    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feat3 = super().forward(x)
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

class FCN16(FCN):

    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feat3 = super().forward(x)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])


class FCN32(FCN):

    def forward(self, x):
        feat3 = super().forward(x)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score = self.score_fconn(fconn)

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


class PSPConv(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, features, 1, bias=False),
            nn.BatchNorm2d(features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=.95),
        )

    def forward(self, x):
        return self.conv(x)


class PSPCross(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.crossa = nn.Sequential(
            nn.Conv2d(in_channels, features, 1, bias=False),
            nn.BatchNorm2d(features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=.95),
        )
        self.crossb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=.95),
        )

    def forward(self, x):
        crossa = self.crossa(x)
        crossb = self.crossb(x)

        return F.relu(crossa + crossb, inplace=True)


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
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

        self.layer2 = PSPCross(128, 64, 256)
        self.layer2a = PSPConv(256, 64, 256)
        self.layer2b = PSPConv(256, 64, 256)

        self.layer3 = PSPCross(256, 128, 512)
        self.layer3a = PSPConv(512, 128, 512)
        self.layer3b = PSPConv(512, 128, 512)
        self.layer3c = PSPConv(512, 128, 512)

        self.layer4 = PSPCross(512, 256, 1024)
        self.layer4a = PSPConv(1024, 256, 1024)
        self.layer4b = PSPConv(1024, 256, 1024)
        self.layer4c = PSPConv(1024, 256, 1024)
        self.layer4d = PSPConv(1024, 256, 1024)
        self.layer4e = PSPConv(1024, 256, 1024)
        self.layer4f = PSPConv(1024, 256, 1024)
        self.layer4g = PSPConv(1024, 256, 1024)
        self.layer4h = PSPConv(1024, 256, 1024)
        self.layer4i = PSPConv(1024, 256, 1024)
        self.layer4j = PSPConv(1024, 256, 1024)
        self.layer4k = PSPConv(1024, 256, 1024)
        self.layer4l = PSPConv(1024, 256, 1024)
        self.layer4m = PSPConv(1024, 256, 1024)
        self.layer4n = PSPConv(1024, 256, 1024)
        self.layer4o = PSPConv(1024, 256, 1024)
        self.layer4p = PSPConv(1024, 256, 1024)
        self.layer4q = PSPConv(1024, 256, 1024)
        self.layer4r = PSPConv(1024, 256, 1024)
        self.layer4s = PSPConv(1024, 256, 1024)
        self.layer4t = PSPConv(1024, 256, 1024)
        self.layer4u = PSPConv(1024, 256, 1024)
        self.layer4v = PSPConv(1024, 256, 1024)

        self.layer5 = PSPCross(1024, 512, 2048)
        self.layer5a = PSPConv(2048, 512, 2048)
        self.layer5b = PSPConv(2048, 512, 2048)

        self.layer6a = nn.Sequential(
            nn.AvgPool2d(60, stride=60),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d((60, 60))
        )
        self.layer6b = nn.Sequential(
            nn.AvgPool2d(60, stride=60),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d((60, 60))
        )
        self.layer6c = nn.Sequential(
            nn.AvgPool2d(60, stride=60),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d((60, 60))
        )
        self.layer6d = nn.Sequential(
            nn.AvgPool2d(60, stride=60),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d((60, 60))
        )

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        layer1 = self.layer1(x)

        layer2 = self.layer2(layer1)
        layer2a = F.relu(self.layer2a(layer2) + layer2, inplace=True)
        layer2b = F.relu(self.layer2b(layer2a) + layer2a, inplace=True)

        layer3 = self.layer3(layer2b)
        layer3a = F.relu(self.layer3a(layer3) + layer3, inplace=True)
        layer3b = F.relu(self.layer3b(layer3a) + layer3a, inplace=True)
        layer3c = F.relu(self.layer3c(layer3b) + layer3b, inplace=True)

        layer4 = self.layer4(layer3c)
        layer4a = F.relu(self.layer4a(layer4) + layer4, inplace=True)
        layer4b = F.relu(self.layer4b(layer4a) + layer4a, inplace=True)
        layer4c = F.relu(self.layer4c(layer4b) + layer4b, inplace=True)
        layer4d = F.relu(self.layer4d(layer4c) + layer4c, inplace=True)
        layer4e = F.relu(self.layer4e(layer4d) + layer4d, inplace=True)
        layer4f = F.relu(self.layer4f(layer4e) + layer4e, inplace=True)
        layer4g = F.relu(self.layer4g(layer4f) + layer4f, inplace=True)
        layer4h = F.relu(self.layer4h(layer4g) + layer4g, inplace=True)
        layer4i = F.relu(self.layer4i(layer4h) + layer4h, inplace=True)
        layer4j = F.relu(self.layer4j(layer4i) + layer4i, inplace=True)
        layer4k = F.relu(self.layer4k(layer4j) + layer4j, inplace=True)
        layer4l = F.relu(self.layer4l(layer4k) + layer4k, inplace=True)
        layer4m = F.relu(self.layer4m(layer4l) + layer4l, inplace=True)
        layer4n = F.relu(self.layer4n(layer4m) + layer4m, inplace=True)
        layer4o = F.relu(self.layer4o(layer4n) + layer4n, inplace=True)
        layer4p = F.relu(self.layer4p(layer4o) + layer4o, inplace=True)
        layer4q = F.relu(self.layer4q(layer4p) + layer4p, inplace=True)
        layer4r = F.relu(self.layer4r(layer4q) + layer4q, inplace=True)
        layer4s = F.relu(self.layer4s(layer4r) + layer4r, inplace=True)
        layer4t = F.relu(self.layer4t(layer4s) + layer4s, inplace=True)
        layer4u = F.relu(self.layer4u(layer4t) + layer4t, inplace=True)
        layer4v = F.relu(self.layer4v(layer4u) + layer4u, inplace=True)

        layer5 = self.layer5(layer4v)
        layer5a = F.relu(self.layer5a(layer5) + layer5, inplace=True)
        layer5b = F.relu(self.layer5b(layer5a) + layer5a, inplace=True)

        final = self.final(torch.cat([
            self.layer6a(layer5b),
            self.layer6b(layer5b),
            self.layer6c(layer5b),
            self.layer6d(layer5b),
        ], 1))

        return F.upsample_bilinear(final, x.size()[2:])