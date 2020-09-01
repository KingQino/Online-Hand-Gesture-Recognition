# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 7:54 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : resnext.py
# @Software: PyCharm
import math

import torch.nn as nn


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channel, out_channel, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_channel = cardinality * int(out_channel / 32)
        self.conv1 = nn.Conv3d(in_channel, mid_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channel)
        self.conv2 = nn.Conv3d(mid_channel, mid_channel, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channel)
        self.conv3 = nn.Conv3d(mid_channel, out_channel * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, blocks_num, sample_size, sample_duration, cardinality=32, num_classes=27,
                 include_top=True):
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # self.conv1 = nn.Conv3d(3, self.in_channel, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.conv1 = nn.Conv3d(3, self.in_channel, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, blocks_num[0], cardinality)
        self.layer2 = self._make_layer(block, 256, blocks_num[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, blocks_num[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, blocks_num[3], cardinality, stride=2)
        if self.include_top:
            last_duration = int(math.ceil(sample_duration / 16))
            last_size = int(math.ceil(sample_size / 32))
            self.avgpool = nn.AvgPool3d(
                (last_duration, last_size, last_size), stride=1)
            self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, block_num, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, cardinality, stride=stride, downsample=downsample))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], **kwargs)
    return model

