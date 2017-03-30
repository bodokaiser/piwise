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


class AdvancedCNN(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels*24, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels*24, num_channels*64, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels*64, num_classes*24, 6, padding=1)
        self.conv4 = nn.Conv2d(num_classes*24, num_classes, 1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return self.conv3(x)


class UNetConv(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = F.relu(self.conv1(inputs))
        outputs = F.relu(self.conv2(outputs))

        return outputs


class UNetEncode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = UNetConv(in_size, out_size)
        self.down = nn.MaxPool2d(2, 1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)

        return outputs


class UNetDecode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = UNetConv(in_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2*[offset // 2, offset // 2 + 1]

        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class UNet(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.encode1 = UNetEncode(num_channels, 32*num_channels)
        self.encode2 = UNetEncode(32*num_channels, 64*num_channels)
        self.encode3 = UNetEncode(128*num_channels, 256*num_channels)
        self.center = UNetConv(256*num_channels, 512*num_channels)
        self.decode3 = UNetDecode(512*num_channels, 256*num_channels)
        self.decode2 = UNetDecode(256*num_channels, 128*num_channels)
        self.decode1 = UNetDecode(64*num_channels, 32*num_channels)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, inputs):
        encode1 = self.encode1(inputs)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        center = self.center(encode3)
        decode3 = self.decode3(encode3, center)
        decode2 = self.decode2(encode2, decode3)
        decode1 = self.decode1(encode1, decode2)

        return self.final(decode1)