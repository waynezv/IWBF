#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import errno
import sys
import random
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
from model.model import _E, _G, _D, weights_init
from model.utils import save_checkpoint, ScoreMeter

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
if args.resume:  # Resume from saved checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        old_args = checkpoint['args']
        print('Old args:')
        print(old_args)

        print("=> creating model")
        netE = _E().cuda()
        netG = _G().cuda()
        netD = _D().cuda()

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
    netE = _E().cuda()
    netG = _G().cuda()
    netD = _D().cuda()
    # netE = torch.nn.DataParallel(netE, device_ids=[0, 1, 2, 3]).cuda()

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
tasks = ['id', 'gender', 'dialect', 'age', 'height']
loader_args = {'batch': True, 'batch_size': args.batchSize, 'shuffle': True, 'num_workers': 32}
train_loader, test_loader = dataloader(featdir, trainlist, testlist, timitinfo, tasks, loader_args)

# Evaluate model
if args.eval:
    print("=> evaluating model")
    if not os.path.exists(os.path.join(args.outf, 'eval')):
        os.makedirs(os.path.join(args.outf, 'eval'))

    print('Done.')
    sys.exit(0)

# Setup optimizer
optimizer_E = optim.Adam(netE.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_G = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))

# Records
if not os.path.exists(os.path.join(args.outf, 'records')):
    os.makedirs(os.path.join(args.outf, 'records'))
configure(os.path.join(args.outf, 'records'))

lossE_meter = ScoreMeter()
lossG_meter = ScoreMeter()
lossD_meter = ScoreMeter()
test_err_meter = ScoreMeter()

# Training settings
save_freqency = 1  # save every epoch
print('Save frequency: ', save_freqency)
old_record_fn = 'youll_never_find_me'
best_test_error = 1e19
best_epoch = 0

'''
# Train model
print("=> traning")
for epoch in range(args.niter):
    data_iter = iter(train_loader)
    i = 0  # data counter
    for x, y in train_loader:
        i += 1

        netE.train()
        netE.zero_grad()
        x = Variable(x.cuda())
        y = Variable(y.float().cuda()).long()

        print('[{:d}/{:d}][{:d}/{:d}][{:d}] '.format(epoch, args.niter, i, len(train_loader), gen_iterations) +
              Fore.RED + 'LossD: {:.4f} '.format(lossD.data[0]) + Fore.RESET +
              'ErrD_real: {:.4f} ErrD_fake: {:.4f} ErrD_grad: {:.4f} '.format(errD_real.data[0], errD_fake.data[0], errD_grad.data[0]) +
              Fore.BLUE + 'LossE: {:.4f} LossP: {:.4f} LossScatter: {:.4f} '.format(lossE.data[0], lossP.data[0], lossScatter.data[0]) + Fore.RESET +
              Fore.GREEN + 'LossG: {:.4f}'.format(lossG.data[0]) + Fore.RESET)

        # Save images & test
        if (gen_iterations % save_freqency) == 0:  # every 5 epochs

            # Test

            # Save best
            is_best = pred_error < best_test_error
            if is_best:
                best_test_error = pred_error
                best_epoch = epoch
                best_generation = gen_iterations
                save_checkpoint({
                    'args': args,
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'best_generation': best_generation,
                    'best_test_error': best_test_error,
                    'netE_state_dict': netE.state_dict(),
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict()
                }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_BEST.pth.tar')
                print(Fore.GREEN + 'Saved checkpoint for best test error {:.4f} at epoch {:d}'.format(best_test_error, best_epoch) + Fore.RESET)

            # Logging
            # log_value('test_err', pred_error, gen_iterations)
            log_value('lossD', lossD.data[0], gen_iterations)
            log_value('lossE', lossE.data[0], gen_iterations)
            log_value('lossG', lossG.data[0], gen_iterations)
            # test_err_meter.update(pred_error)
            lossD_meter.update(lossD.data[0])
            lossE_meter.update(lossE.data[0])
            lossG_meter.update(lossG.data[0])

            # Checkpointing
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'netE_state_dict': netE.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict()
            }, os.path.join(args.outf, 'checkpoints'), 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))

            # Delete old checkpoint to save space
            new_record_fn = os.path.join(args.outf, 'checkpoints', 'checkpoint_gen_{:d}_epoch_{:d}.pth.tar'.format(gen_iterations, epoch))
            if os.path.exists(old_record_fn) and os.path.exists(new_record_fn):
                os.remove(old_record_fn)
            old_record_fn = new_record_fn

# Write log
test_err_meter.save('test_err', os.path.join(args.outf, 'records'), 'test_err.tsv')
lossD_meter.save('lossD', os.path.join(args.outf, 'records'), 'lossD.tsv')
lossE_meter.save('lossE', os.path.join(args.outf, 'records'), 'lossE.tsv')
lossG_meter.save('lossG', os.path.join(args.outf, 'records'), 'lossG.tsv')
'''
