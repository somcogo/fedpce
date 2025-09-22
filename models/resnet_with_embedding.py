import math

import torch
from torch import nn

from .embedding_functionals import EmbNormLayer

# Based on the official PyTorch ResNet-18 implementation
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html
class CustomResnet(nn.Module):
    def __init__(self, channels=3, layers=[2, 2, 2, 2], feature_dims=[64, 128, 256, 512], cifar=False, comb_gen_length=0, **kwargs):
        super().__init__()
        self.comb_gen_length = comb_gen_length
        self.kwargs = kwargs

        if cifar:
            self.conv1 = nn.Conv2d(channels, feature_dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(channels, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = EmbNormLayer(num_features=feature_dims[0], **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(feature_dims[0], feature_dims[0], layers[0], **kwargs)
        self.layer2 = self.make_layer(feature_dims[0], feature_dims[1], layers[1], stride=2, **kwargs)
        self.layer3 = self.make_layer(feature_dims[1], feature_dims[2], layers[2], stride=2, **kwargs)
        self.layer4 = self.make_layer(feature_dims[2], feature_dims[3], layers[3], stride=2, **kwargs)


    def make_layer(self, in_channels, out_channels, depth, stride=1, **kwargs):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                EmbNormLayer(out_channels, **kwargs)
            )
        layers = []
        layers.append(ResnetBlock(in_channels, out_channels, stride, downsample, **kwargs))

        for _ in range(1, depth):
            layers.append(ResnetBlock(out_channels, out_channels, **kwargs))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return [feat1, feat2, feat3, feat4]

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = EmbNormLayer(num_features=out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = EmbNormLayer(num_features=out_channels, **kwargs)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out