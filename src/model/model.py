#!/usr/bin/env python
# encoding: utf-8

import math
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import pdb


class _E(nn.Module):
    '''
    Encoder.
    '''
    def __init__(self):
        super(_E, self).__init__()
        # x 1*414*450
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # 200
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 100
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 50
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 25
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # 12
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1024*1*1
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 200, 1, 1, 0, bias=False)  # 200*1*1
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class _G(nn.Module):
    '''
    Decoder / Generator.
    '''
    def __init__(self):
        super(_G, self).__init__()
        # z 200*1*1
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(200, 1024, 16, 2, 0, bias=False),  # 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # 512
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d((414, 450))  # 1*414*450
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class _D(nn.Module):
    '''
    Discriminator.

    '''
    def __init__(self):
        super(_D, self).__init__()
        # z 200*1*1
        self.classifier = nn.Sequential(
            nn.Conv2d(200, 1000, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1000, 1000, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1000, 1, 1, 1, 0)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1)


def weights_init(m):
    '''
    Custom weights initialization.
    '''
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
        m.bias.data.zero_()
