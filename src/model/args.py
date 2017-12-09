#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='timit | tidigits | to be added')

parser.add_argument('--batchSize', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for. default=30')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--outf', default='', help='folder to save results and model checkpoints')

# optimizer
parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd, momentum, nesterov, adagrad, adadelta, rmsprop, adam. default=sgd')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate. default=0.001')

# accelerate
parser.add_argument('--momentum', type=float, default=0.5, help='momentum. default=0.5')
parser.add_argument('--nesterov', action='store_true', default=False, help='nesterov. default=False')

# adadelta
parser.add_argument('--rho', type=float, default=0.9, help='rho for adadelta. default=0.9')

# rmsprop
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for rmsprop. default=0.9')

# adam
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')

# Resume settings
parser.add_argument('--resume', default='', help='path of saved checkpoint to resume from')
parser.add_argument('--eval', action='store_true', help='evaluate trained model')
