import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x): 
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.leaky_relu(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.res1 = ResidualBlock(in_channels=6, out_channels=32)
        self.res2 = ResidualBlock(in_channels=32, out_channels=64)
        #self.res3 = ResidualBlock(in_channels=64, out_channels=128)

        #self.res4 = ResidualBlock(in_channels=128, out_channels=64)
        self.res5 = ResidualBlock(in_channels=64, out_channels=32)
        self.res6 = ResidualBlock(in_channels=32, out_channels=16)
        self.conv_out = nn.Conv2d(16, 3, kernel_size=1, stride=1)

    def forward(self, x ): 
        x1 = self.res1(x)  # (6, 256, 256) -> (32, 256, 256)
        x2 = self.res2(x1)  # (32, 256, 256) -> (64, 256, 256)
        #x3 = self.res3(x2)  # (64, 256, 256) -> (128, 256, 256)

        #x4 = self.res4(x3)  # (128, 256, 256) -> (64, 256, 256)
        x5 = self.res5(x2)  # (64, 256, 256) -> (32, 256, 256)
        x6 = self.res6(x5)  # (32, 256, 256) -> (16, 256, 256)

        out = torch.sigmoid(self.conv_out(x6))  # (16, 256, 256) -> (3, 256, 256)
        return out
