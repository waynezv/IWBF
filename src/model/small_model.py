#!/usr/bin/env python
# encoding: utf-8

import math
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb


class _E(nn.Module):
    '''
    Encoder.
    '''
    def __init__(self, attri_dict):
        super(_E, self).__init__()
        self.attri_dict = attri_dict
        # x 1*414*450
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((512, 512)),  # 1*512*512
            nn.Conv2d(1, 16, 4, 4, 0, bias=False),  # 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 4, 0, bias=False),  # 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 4, 0, bias=False),  # 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 8, 8, 0, bias=False),  # 128*1*1

        )
        self.demux= nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 1, 1, 0)
        )
        self.encoders = nn.ModuleList()
        self.encoders.append(self.demux)  # for decoder
        for k in self.attri_dict:
            self.encoders.append(self.demux)

    def forward(self, x):
        x = self.encoder(x)
        zs = OrderedDict()
        zs['encode'] = self.encoders[0](x)
        for i, k in enumerate(self.attri_dict):
            zs[k] = self.encoders[i + 1](x)
        return zs


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
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d((414, 450)),  # 1*414*450
            nn.ConvTranspose2d(1, 1, 3, 1, 1),
            nn.BatchNorm2d(1)  # use batchnorm if have normalized data
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
        self.logvar = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0)
        )
        self.projector = nn.Sequential(
            nn.Conv2d(128, 1024, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
            # nn.Conv2d(1024, 1024, 1, 1, 0),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.5)
        )
        self.projectors = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for k in self.attri_dict:
            self.projectors.append(self.projector)
            self.classifiers.append(nn.Linear(1024, self.attri_dict[k]['dimension']))

    def sample(self, x):
        mean = x
        logvar = self.logvar(x)
        var = logvar.exp()
        eps = torch.normal(torch.zeros(*var.size())).cuda()  # N(0, I)
        # Reparameterization trick
        z = mean + var * Variable(eps, requires_grad=False)  # var is std
        return z

    def forward(self, x):
        # h = self.projector(x).view(-1, 1024)
        ys = OrderedDict()
        for i, k in enumerate(self.attri_dict):
            # z = self.sample(x)
            h = self.projectors[i](x[k]).view(-1, 1024)
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
