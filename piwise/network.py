import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 256, 3, 1, 1)

    def forward(self, image):
        return self.conv(image)

class UNetConv(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

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

    def __init__(self):
        super().__init__()

        self.encode1 = UNetEncode(1, 64)
        self.encode2 = UNetEncode(64, 128)
        self.encode3 = UNetEncode(128, 256)
        self.encode4 = UNetEncode(256, 512)
        self.center = UNetConv(512, 1024)
        self.decode4 = UNetDecode(1024, 512)
        self.decode3 = UNetDecode(512, 256)
        self.decode2 = UNetDecode(256, 128)
        self.decode1 = UNetDecode(128, 64)
        self.final = nn.Conv2d(64, 256, 1)

    def forward(self, inputs):
        encode1 = self.encode1(inputs)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        center = self.center(encode4)
        decode4 = self.decode4(encode4, center)
        decode3 = self.decode3(encode3, decode4)
        decode2 = self.decode2(encode2, decode3)
        decode1 = self.decode1(encode1, decode2)

        return self.final(decode1)
