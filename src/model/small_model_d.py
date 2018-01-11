#!/usr/bin/env python
# encoding: utf-8

from collections import OrderedDict
import torch.nn as nn
import pdb


class _D(nn.Module):
    '''
    Discriminator.

    attri_dict: OrderedDict, {label name: {discrete: True or False, dimension: number of classes [int]}}
    '''
    def __init__(self, attri_dict):
        super(_D, self).__init__()
        self.attri_dict = attri_dict
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
        self.projector = nn.Sequential(
            nn.Conv2d(128, 1024, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
            # nn.Conv2d(1024, 1024, 1, 1, 0),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.5)
        )
        self.classifiers = nn.ModuleList()
        for k in self.attri_dict:
            self.classifiers.append(nn.Linear(1024, self.attri_dict[k]['dimension']))

    def forward(self, x):
        z = self.encoder(x)
        h = self.projector(z).view(-1, 1024)
        ys = OrderedDict()
        for i, k in enumerate(self.attri_dict):
            ys[k] = self.classifiers[i](h)
        return ys
