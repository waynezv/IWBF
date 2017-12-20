#!/usr/bin/env python
# encoding: utf-8

import math
from collections import OrderedDict
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
            nn.AdaptiveAvgPool2d((512, 512)),  # 1*512*512
            nn.Conv2d(1, 16, 4, 4, 0, bias=False),  # 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 4, 0, bias=False),  # 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 4, 0, bias=False),  # 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 8, 8, 0, bias=False)  # 128*1*1
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
        # z 128*1*1
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 8, 8, 0, bias=False),  # 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 4, 0, bias=False),  # 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 4, 0, bias=False),  # 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, 4, 4, 0, bias=False),  # 1*512*512
            nn.AdaptiveMaxPool2d((414, 450))  # 1*414*450
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class _D(nn.Module):
    '''
    Discriminator.

    '''
    def __init__(self, attri_dict):
        '''
        attri_dict: OrderedDict, {label name: {discrete: True or False, dimension: number of classes [int]}}
        '''
        super(_D, self).__init__()
        # z 128*1*1
        self.attri_dict = attri_dict
        self.projector = nn.Sequential(
            nn.Conv2d(128, 1024, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.LeakyReLU(0.2)
        )
        self.classifiers = nn.ModuleList()
        for k in self.attri_dict:
            self.classifiers.append(nn.Linear(1024, self.attri_dict[k]['dimension']))

    def forward(self, x):
        h = self.projector(x).view(-1, 1024)
        ys = OrderedDict()
        for i, k in enumerate(self.attri_dict):
            ys[k] = self.classifiers[i](h)
        return ys


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
