# -*- coding: utf-8 -*-
# @Time    : 2020/7/19 10:41 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : c3d.py
# @Software: PyCharm
import math

import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes, sample_duration):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        last_duration = int(math.floor(sample_duration / 16))
        self.fc6 = nn.Linear(8192 * last_duration, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)

        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)

        h = self.relu(self.bn3a(self.conv3a(h)))
        h = self.relu(self.bn3b(self.conv3b(h)))
        h = self.pool3(h)

        h = self.relu(self.bn4a(self.conv4a(h)))
        h = self.relu(self.bn4b(self.conv4b(h)))
        h = self.pool4(h)

        h = self.relu(self.bn5a(self.conv5a(h)))
        h = self.relu(self.bn5b(self.conv5b(h)))
        h = self.pool5(h)

        h = h.view(h.size(0), -1)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)

        return logits