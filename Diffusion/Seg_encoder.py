import torch
import torch.nn as nn
import torch.nn.functional as F


    

class SegEncoder(nn.Module):
    def __init__(self):
        super(SegEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv_out = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=1,stride=1)
        
    def forward(self, x ):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))  # (6, 256, 256) -> (16, 256, 256)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))  # (16, 256, 256) -> (32, 256, 256)
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)))
        out = torch.tanh(self.conv_out(x3))
        return out
