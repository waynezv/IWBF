#!/usr/bin/env python
# encoding: utf-8

import math
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb


class ResBlock(nn.Module):
    '''
    Resnet block.
    '''
    def __init__(self, inplanes):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(inplanes * 2),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(inplanes * 2, inplanes * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(inplanes * 4)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, inplanes * 4, 4, 4, 0, bias=False),
            nn.BatchNorm2d(inplanes * 4)
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.block1(x)
        out = self.block2(out)

        out += residual
        out = self.relu(out)

        return out


class ResTransBlock(nn.Module):
    '''
    Transposed Resnet block.
    '''
    def __init__(self, inplanes):
        super(ResTransBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(inplanes, inplanes / 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(inplanes / 2, inplanes / 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(inplanes / 4)
        )
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(inplanes, inplanes / 4, 4, 4, 0, bias=False),
            nn.BatchNorm2d(inplanes / 4)
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.block1(x)
        out = self.block2(out)

        out += residual
        out = self.relu(out)

        return out


class _E(nn.Module):
    '''
    Encoder.
    '''
    def __init__(self, block):
        super(_E, self).__init__()
        # x 1*414*450
        self.prelayers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((256, 256))  # 16*256*256
        )

        self.layers = nn.Sequential(
            block(16),  # 64*64*64
            block(64)  # 256*16*16
        )

        self.postlayers = nn.Sequential(
            nn.Conv2d(256, 1024, 4, 4, 0),  # 1024*4*4
            nn.BatchNorm2d(1024),  # constrain latent space to normal
            nn.AvgPool2d(4, 4, 0)  # 1024*1*1
        )

    def forward(self, x):
        out = self.prelayers(x)

        out = self.layers(out)

        out = self.postlayers(out)

        return out


class _G(nn.Module):
    '''
    Decoder / Generator.
    '''
    def __init__(self, block):
        super(_G, self).__init__()
        # z 1024*1*1
        self.prelayers = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 4, 0, bias=False),  # 1024*4*4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 256, 4, 4, 0, bias=False),  # 256*16*16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.Sequential(
            block(256),  # 64*64*64
            block(64)  # 16*256*256
        )

        self.postlayers = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 4, 2, 1, bias=False),  # 1*512*512
            nn.BatchNorm2d(1),  # use batchnorm if have normalized input
            nn.AdaptiveAvgPool2d((414, 450))  # 1*414*450
        )

    def forward(self, x):
        out = self.prelayers(x)

        out = self.layers(out)

        out = self.postlayers(out)

        return out


class _D(nn.Module):
    '''
    Discriminator.

    '''
    def __init__(self, attri_dict):
        '''
        attri_dict: OrderedDict, {label name: {discrete: True or False, dimension: number of classes [int]}}
        '''
        super(_D, self).__init__()
        # z 1024*1*1
        self.attri_dict = attri_dict
        self.projector = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
        self.projectors = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for k in self.attri_dict:
            self.projectors.append(self.projector)
            self.classifiers.append(nn.Linear(1024, self.attri_dict[k]['dimension']))

    def forward(self, x):
        ys = OrderedDict()
        for i, k in enumerate(self.attri_dict):
            h = self.projectors[i](x).view(-1, 1024)
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
