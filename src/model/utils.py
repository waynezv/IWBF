from __future__ import print_function
import sys
import time
import os
import shutil
from collections import OrderedDict
import torch
import numpy as np
from colorama import Fore
import pdb


def loss_func(yps, y, attri_dict):
    '''
    Compute losses w.r.t. attri_dict.

    :yps: OrderedDict, {label name: prediction (without softmax) [TorchTensor]}
    :y: TorchTensor, label
    :attri_dict: OrderedDict, {label name: {discrete: True or False, dimension: number of classes [int]}}
    <- losses: OrderedDict, {label name: loss}
    '''
    losses = OrderedDict()
    for i, k in enumerate(attri_dict):
        if attri_dict[k]['discrete'] is True:  # discrete classes
            losses[k] = torch.nn.functional.cross_entropy(yps[k], y[:, i].long())
        else:  # continuous values
            losses[k] = torch.nn.functional.mse_loss(yps[k], y[:, i])

    return losses


def cal_test_err(loss_ds, yps, y, attri_dict):
    '''
    Compute test errors w.r.t. attri_dict.

    :loss_ds: OrderedDict, {label name: loss [TorchTensor]}
    :yps: OrderedDict, {label name: prediction (without softmax) [TorchTensor]}
    :y: TorchTensor, label
    :attri_dict: OrderedDict, {label name: {discrete: True or False, dimension: number of classes [int]}}
    <- test_errs: OrderedDict, {label name: test error [float]]}
    '''
    test_errs = OrderedDict()
    for i, k in enumerate(attri_dict):
        if attri_dict[k]['discrete'] is True:  # discrete classes
            test_errs[k] = torch.nn.functional.softmax(yps[k]).topk(1, dim=1)[1].squeeze().eq(y[:, i].long()).sum().data[0] / float(y.size(0))
        else:  # continuous values
            test_errs[k] = loss_ds[k].data[0]

    return test_errs


class ScoreMeter():
    '''
    Record scores.
    '''
    def __init__(self):
        self.score = []

    def update(self, val):
        self.score.append(val)

    def save(self, score_name, save_dir, fn):
        scores = "idx\t{}".format(score_name)
        for i, s in enumerate(self.score):
            scores += "\n"
            scores += "{:d}\t{:.4f}".format(i, s)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fn = os.path.join(save_dir, fn)
        with open(fn, 'w') as f:
            f.write(scores)


class AverageMeter(object):
    '''
    Computes and stores the average and current value.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    '''
    Create new folder and backup old folder.
    '''
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET +
              ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED +
                                              tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def save_checkpoint(state, save_dir, filename, is_best=False):
    '''
    Save training checkpoints.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    '''
    Return optimizer.
    '''
    if args.optimizer == 'sgd':
        print('got sgd')
        return torch.optim.SGD(model.parameters(),
                               lr=args.lr, momentum=0., nesterov=False)

    if args.optimizer == 'momentum':
        print('got sgd with momentum {}'.format(args.momentum))
        return torch.optim.SGD(model.parameters(),
                               lr=args.lr, momentum=args.momentum, nesterov=False)

    if args.optimizer == 'nesterov':
        print('got sgd with momentum {} and nesterov {}'.format(args.momentum, args.nesterov))
        return torch.optim.SGD(model.parameters(),
                               lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)

    elif args.optimizer == 'adagrad':
        print('got adagrad with lr {}'.format(args.lr))
        return torch.optim.Adadelta(model.parameters(), lr=args.lr)

    elif args.optimizer == 'adadelta':
        print('got adadelta with rho {}'.format(args.rho))
        return torch.optim.Adadelta(model.parameters(), rho=args.rho)

    elif args.optimizer == 'rmsprop':
        print('got rmsprop with alpha {}'.format(args.alpha))
        return torch.optim.RMSprop(model.parameters(),
                                   lr=args.lr, alpha=args.alpha)

    elif args.optimizer == 'adam':
        print('got adam with beta1 {} and beta2 {}'.format(args.beta1, args.beta2))
        return torch.optim.Adam(model.parameters(),
                                lr=args.lr, betas=(args.beta1, args.beta2))

    else:
        raise NotImplementedError


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    '''
    Decay Learning rate at 1/2 and 3/4 of the num_epochs.
    '''
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
