from torch import nn
import torch.nn.functional as F
import torch

from .layers import *


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(
        self,
        inchannels: int,
        outchannels: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inchannels, inchannels)
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.conv2 = conv3x3(inchannels, inchannels, stride)
        self.bn2 = nn.BatchNorm2d(inchannels)
        self.conv3 = conv1x1(inchannels, outchannels)
        self.bn3 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inchannels != outchannels:
            self.downsample = nn.Sequential(
                conv1x1(inchannels, outchannels, stride),
                nn.BatchNorm2d(outchannels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, channel):
        super().__init__()
        self.up = WRC(
            in_channels=in_channels//2,
            out_channels=in_channels//2,
            kernel_size=2,
            scale=2,
            padding=2,
            padding_mode='circular', 
            eps=1e-5
        )
        self.conv = nn.Sequential(
            Bottleneck(in_channels, in_channels//2, 1),
            Bottleneck(in_channels//2, in_channels//2, 1)
        )
        self.conv_1 = nn.Sequential(
            Bottleneck(in_channels//2+channel, out_channels//2, 1),
        )

    def forward(self, x, imgs_1, ouput_size=None):
        x = self.conv(x)
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


class WRCUpsampler(nn.Module):
    def __init__(self, in_channels, patch_size, pre_shape=True, post_shape=True, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        self.channel = 32
        
        self.up1 = (Up(in_channels+self.channel, in_channels, self.channel))
        self.outc = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.in_conv = nn.Sequential(
            conv1x1(3, self.channel),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(inplace=True),
        )
        self.image_convs_1 = nn.Sequential(
            Bottleneck(self.channel, self.channel, 2),
            Bottleneck(self.channel, self.channel, 1),
            Bottleneck(self.channel, self.channel, 2),
            Bottleneck(self.channel, self.channel, 1),
        )
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        elif patch_size == 16:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        elif patch_size == 14:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        else:
            print('ERROR: patch size %i not currently supported'%patch_size)
            exit()
        self.image_convs_2 = nn.Sequential(
            Bottleneck(self.channel, self.channel, 2),
            Bottleneck(self.channel, self.channel, 1),
        )


    # [B, T, C] --> [B, C, H, W]
    def run_pre_shape(self, imgs, x):
        H = int(imgs.shape[2] / self.patch_size)
        W = int(imgs.shape[3] / self.patch_size)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, H, W)
        return x


    # [B, C, H, W] --> [B, T, C]
    def run_post_shape(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


    def forward(self, imgs, x, output_size=None, **kwargs):
        output_size = output_size if output_size is not None else imgs.shape[-2:]
        if self.pre_shape: x = self.run_pre_shape(imgs, x)
        imgs_1 = self.in_conv(imgs)
        imgs_1 = self.image_convs_1(imgs_1)
        if self.patch_size == 14:
            imgs_1 = F.interpolate(imgs_1, (x.shape[-2]*4, x.shape[-1]*4), mode='bilinear', align_corners=False)
        imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        # Enable the following if working with both --imsize 56 and --patch_size 16
        # if(x.shape[2] != imgs_2.shape[2]):
        #     imgs_1 = self.image_convs_1(imgs[:,:,2:-2,2:-2])
        #     imgs_1 = self.scale_adapter(imgs_1)
        #     imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        if self.post_shape: logits = self.run_post_shape(logits)

        return logits
