#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import errno
import sys
import random
from collections import OrderedDict
import pdb

import numpy as np
from scipy.io import savemat

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from tensorboard_logger import configure, log_value
from colorama import Fore

from model.args import parser
from model.dataloader import dataloader
# from model.model import _E, _G, _D, weights_init
# from model.small_model import _E, _G, _D, weights_init
from model.resnet_model import ResBlock, ResTransBlock, _E, _G, _D, weights_init
from model.utils import save_checkpoint, ScoreMeter, loss_func, cal_err, taskwise_earlystop

# Parse args
args = parser.parse_args()
print(args)

# Make dirs
try:
    os.makedirs(args.outf)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Fix seed for randomization
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
rng = np.random.RandomState(seed=args.manualSeed)

if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# CUDA, CUDNN
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# cudnn.benchmark = True
cudnn.fastest = True

# Init model
tasks = ['id', 'gender', 'dialect', 'age', 'height']
# tasks = ['age']
# tasks = []
attri_dict = OrderedDict()
attri_dict['id'] = {'discrete': True, 'dimension': 630}
attri_dict['gender'] = {'discrete': True, 'dimension': 2}
attri_dict['dialect'] = {'discrete': True, 'dimension': 8}
attri_dict['age'] = {'discrete': False, 'dimension': 1}
attri_dict['height'] = {'discrete': False, 'dimension': 1}
print(Fore.GREEN + 'Tasks: ' + Fore.RESET, end='')
print(tasks)
loss_weight_dict = OrderedDict()
loss_weight_dict['reconstruction'] = 1.
loss_weight_dict['id'] = 1.
loss_weight_dict['gender'] = 1.
loss_weight_dict['dialect'] = 1.
loss_weight_dict['age'] = 1.
loss_weight_dict['height'] = 1.
print(Fore.GREEN + 'Tasks weights: ' + Fore.RESET, end='')
print(loss_weight_dict)

if args.resume:  # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        old_args = checkpoint['args']
        print('Old args:')
        print(old_args)

        print("=> creating model")
        netE = _E(ResBlock).cuda()
        netG = _G(ResTransBlock).cuda()
        netD = _D(attri_dict).cuda()

        netE.load_state_dict(checkpoint['netE_state_dict'])
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        print("=> loaded model with checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'"
              .format(Fore.RED + args.resume + Fore.RESET), file=sys.stderr)
        sys.exit(0)

else:  # Create model from scratch
    print("=> creating model")
    netE = _E(ResBlock).cuda()
    netG = _G(ResTransBlock).cuda()
    netD = _D(attri_dict).cuda()

    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)

print(netE)
print(netG)
print(netD)

# Prepare data
featdir = '../data/timit/sentence_constq_feats'
trainlist = './ctls/train_sentence_constq.ctl'
testlist = './ctls/test_sentence_constq.ctl'
timitinfo = './ctls/timit.info'
print('=> loading data')
loader_args = {'num_train': args.numTrain, 'num_test': args.numTest,
               'batch': True, 'batch_size': args.batchSize,
               'shuffle': True, 'num_workers': 32}
train_loader, test_loader = dataloader(featdir, trainlist, testlist, timitinfo, tasks, **loader_args)

# Evaluate model
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    print('Done.')
    sys.exit(0)

# Setup optimizer
optimizer_E = optim.Adam(netE.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.001, betas=(0.9, 0.999))

# Records
if not os.path.exists(os.path.join(args.outf, 'records')):
    os.makedirs(os.path.join(args.outf, 'records'))
configure(os.path.join(args.outf, 'records'))

# Training settings
save_freqency = 1  # save every epoch
print('Save frequency: ', save_freqency)
old_record_fn = 'youll_never_find_me'
best_test_error = 1e19
best_epoch = 0
Giters = 1  # num updates for AE

train_losses_dict = OrderedDict()  # save losses for early stop
test_losses_dict = OrderedDict()
stop_stride = 10  # early stop stride
stop_threshold = 100  # early stop threshold
train_losses_dict['reconstruction'] = []
test_losses_dict['reconstruction'] = []
for k in tasks:
    train_losses_dict[k] = []
    test_losses_dict[k] = []

# Train model
print("=> traning")
for epoch in range(args.nepoch):
    data_iter = iter(train_loader)
    i = 0  # data counter
    Gcnt = 0  # AE update counter
    Dcnt = 0  # D update counter

    # Init loss recorder for each epoch
    train_loss_r = 0
    train_loss_ds = OrderedDict()
    for t in tasks:
        train_loss_ds[t] = 0
    train_err_ds = OrderedDict()
    for t in tasks:
        train_err_ds[t] = 0

    while i < len(train_loader) - Giters - 1:  # in case data iter exceeds total count
        # Encoding - decoding
        x, y = data_iter.next()
        i += 1
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        netE.train()
        netG.train()
        netD.train()
        netE.zero_grad()
        netG.zero_grad()
        netD.zero_grad()

        z = netE(x)
        xr = netG(z)

        # loss_r = (xr - x).pow(2).sum(dim=1).sum(dim=1).sum(dim=1).mean()  # + z.norm(p=1)  # Fro-norm square
        loss_r = (xr - x).abs().mean()  # wass: inf{Er[|X - Y|]}
        # loss_r = (xr.mean() - x.mean()).abs()  # dual wass: sup{|Eu[f] - Ev[f]|: ||f||_{L<=1}}
        Gcnt += 1
        train_loss_r += loss_r.data[0]

        # Backprop gradients and update
        if epoch > stop_stride:
            criterion, stop = taskwise_earlystop(train_losses_dict, test_losses_dict, loss_weight_dict, stop_stride, stop_threshold)
            if stop['reconstruction'] is False:
                loss_r.backward(retain_graph=True)
                optimizer_E.step()
                optimizer_G.step()

        else:  # backprop for initial epochs
            loss_r.backward(retain_graph=True)
            optimizer_E.step()
            optimizer_G.step()

        # Discriminating
        x, y = data_iter.next()
        i += 1
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        z = netE(x)
        yps = netD(z)

        loss_ds = loss_func(yps, y, attri_dict, loss_weight_dict)
        Dcnt += 1
        for k in loss_ds:
            train_loss_ds[k] += loss_ds[k].data[0]
        train_err = cal_err(loss_ds, yps, y, attri_dict)  # test errors
        for k in train_err:
            train_err_ds[k] += train_err[k]

        # Backprop gradients and update
        if epoch > stop_stride:
            criterion, stop = taskwise_earlystop(train_losses_dict, test_losses_dict, loss_weight_dict, stop_stride, stop_threshold)

            for k in loss_ds:
                if stop[k] is False:  # not stop
                    loss_ds[k].backward(retain_graph=True)

        else:  # backprop for initial epochs
            for k in loss_ds:
                loss_ds[k].backward(retain_graph=True)

        optimizer_E.step()
        optimizer_D.step()

        print('[{:d}/{:d}][{:d}/{:d}] '.format(epoch, args.nepoch, i, len(train_loader)),
              Fore.RED + 'loss_r: {:.4f} '.format(loss_r.data[0]) + Fore.RESET,
              end='')
        for k in loss_ds:
            print(Fore.BLUE + '{}: {:.4f} '.format(k, loss_ds[k].data[0]) + Fore.RESET, end='')
        print('')

    # Average
    train_loss_r /= float(Gcnt)
    for k in train_loss_ds:
        train_loss_ds[k] /= float(Dcnt)
    for k in train_err_ds:
        train_err_ds[k] /= float(Dcnt)

    # Accumulate losses for stop criterion
    train_losses_dict['reconstruction'].append(train_loss_r)
    for k in train_loss_ds:
        train_losses_dict[k].append(train_loss_ds[k])

    print('[{:d}/{:d}] '.format(epoch, args.nepoch) +
          'train_loss_r: {:.4f} '.format(train_loss_r),
          end='')
    for k in train_loss_ds:
        print('{}: {:.4f} '.format(k, train_loss_ds[k]), end='')
    print('Train error ', end='')
    for k in train_err_ds:
        print('{}: {:.4f} '.format(k, train_err_ds[k]), end='')
    print('')

    # Test
    test_loss_r = 0
    test_loss_ds = OrderedDict()
    for t in tasks:
        test_loss_ds[t] = 0
    test_err_ds = OrderedDict()
    for t in tasks:
        test_err_ds[t] = 0

    for x, y in test_loader:
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        netE.eval()
        netG.eval()
        netD.eval()

        z = netE(x)
        xr = netG(z)

        # loss_r = (xr - x).pow(2).sum(dim=1).sum(dim=1).sum(dim=1).mean()  # + z.norm(p=1)
        loss_r = (xr - x).abs().mean()  # wass: inf{Er[|X - Y|]}
        # loss_r = (xr.mean() - x.mean()).abs()  # dual wass: sup{|Eu[f] - Ev[f]|: ||f||_{L<=1}}
        test_loss_r += loss_r.data[0]

        yps = netD(z)

        loss_ds = loss_func(yps, y, attri_dict, loss_weight_dict)  # test losses
        test_err = cal_err(loss_ds, yps, y, attri_dict)  # test errors
        for k in loss_ds:
            test_loss_ds[k] += loss_ds[k].data[0]
        for k in test_err:
            test_err_ds[k] += test_err[k]

    # Average
    test_loss_r /= float(len(test_loader))
    for k in test_loss_ds:
        test_loss_ds[k] /= float(len(test_loader))
    for k in test_err_ds:
        test_err_ds[k] /= float(len(test_loader))

    # Accumulate losses for stop criterion
    test_losses_dict['reconstruction'].append(test_loss_r)
    for k in test_loss_ds:
        test_losses_dict[k].append(test_loss_ds[k])

    print('[{:d}/{:d}] '.format(epoch, args.nepoch) +
          'test loss_r: {:.4f} '.format(test_loss_r),
          end='')
    for k in test_loss_ds:
        print('{}: {:.4f} '.format(k, test_loss_ds[k]), end='')
    print('test error ', end='')
    for k in test_err_ds:
        print('{}: {:.4f} '.format(k, test_err_ds[k]), end='')
    print('')

    '''
    # Save best
    is_best = pred_error < best_test_error
    if is_best:
        best_test_error = pred_error
        best_epoch = epoch
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_test_error': best_test_error,
            'netE_state_dict': netE.state_dict(),
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict()
        }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST.pth.tar')
        print(Fore.GREEN + 'Saved checkpoint for best test error {:.4f} at epoch {:d}'.format(best_test_error, best_epoch) + Fore.RESET)
    '''

    # Logging
    log_value('train_loss_r', train_loss_r, epoch)
    log_value('train_loss_id', train_loss_ds['id'], epoch)
    log_value('train_loss_gender', train_loss_ds['gender'], epoch)
    log_value('train_loss_dialect', train_loss_ds['dialect'], epoch)
    log_value('train_loss_age', train_loss_ds['age'], epoch)
    log_value('train_loss_height', train_loss_ds['height'], epoch)
    log_value('train_err_id', train_err_ds['id'], epoch)
    log_value('train_err_gender', train_err_ds['gender'], epoch)
    log_value('train_err_dialect', train_err_ds['dialect'], epoch)

    log_value('test_loss_r', test_loss_r, epoch)
    log_value('test_loss_id', test_loss_ds['id'], epoch)
    log_value('test_loss_gender', test_loss_ds['gender'], epoch)
    log_value('test_loss_dialect', test_loss_ds['dialect'], epoch)
    log_value('test_loss_age', test_loss_ds['age'], epoch)
    log_value('test_loss_height', test_loss_ds['height'], epoch)
    log_value('test_err_id', test_err_ds['id'], epoch)
    log_value('test_err_gender', test_err_ds['gender'], epoch)
    log_value('test_err_dialect', test_err_ds['dialect'], epoch)

    if epoch > stop_stride:
        log_value('id', stop['id'], epoch)
        log_value('gender', stop['gender'], epoch)
        log_value('dialect', stop['dialect'], epoch)
        log_value('age', stop['age'], epoch)
        log_value('height', stop['height'], epoch)

        log_value('stop_criterion_id', criterion['id'], epoch)
        log_value('stop_criterion_gender', criterion['gender'], epoch)
        log_value('stop_criterion_dialect', criterion['dialect'], epoch)
        log_value('stop_criterion_age', criterion['age'], epoch)
        log_value('stop_criterion_height', criterion['height'], epoch)

    # Checkpointing
    save_checkpoint({
        'args': args,
        'epoch': epoch,
        'netE_state_dict': netE.state_dict(),
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict()
    }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))

    # Delete old checkpoint to save space
    new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_epoch_{:d}.pth.tar'.format(epoch))
    if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
        os.remove(old_record_fn)
    old_record_fn = new_record_fn

# Write log
# test_err_meter.save('test_err', os.path.join(args.outf, 'records'), 'test_err.tsv')
