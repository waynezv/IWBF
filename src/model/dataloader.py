#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import os
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pdb


def build_info_dict(raw_info_fn):
    '''
    Build dict{spkname: dict{classname: classvalue}}.
    '''
    raw_info = [l.split() for l in open(raw_info_fn)]
    raw_info = raw_info[1:]  # remove header
    info_dict = dict()
    gender_dict = {'f': 0, 'm': 1}  # dictionary for male or female
    spkid_dict = dict()  # dictionary for speaker id
    # race_dict = dict()  # dictionary for race
    # edu_dict = dict()  # dictionary for education

    for l in raw_info:
        spkname = l[0]
        if spkname in spkid_dict:
            spkid = spkid_dict[spkname]
        else:
            spkid_dict[spkname] = len(spkid_dict)
            spkid = spkid_dict[spkname]
        gender = gender_dict[l[1]]
        dialect = int(l[2]) - 1
        birthdate = np.asarray(l[3].split('/'), dtype=np.float64)
        age = (birthdate[0] * 30 + birthdate[1]) / 365. + (86 - birthdate[2])  # age relative to 1986
        h = np.asarray(l[4].rstrip('"').split('\''), dtype=np.float64)
        height = h[0] + h[1] / 12.  # 1 ft = 12 in
        # race = l[5]
        # if race in race_dict:
            # raceid = race_dict[race]
        # else:
            # race_dict[race] = len(race_dict)
            # raceid = race_dict[race]
        # edu = l[6]
        # if edu in edu_dict:
            # eduid = edu_dict[edu]
        # else:
            # edu_dict[edu] = len(edu_dict)
            # eduid = edu_dict[edu]
        info_dict[spkname] = {'id': spkid, 'gender': gender, 'dialect': dialect, 'age': age, 'height': height}
    return info_dict


def dataloader(featdir, trainlist, testlist, timitinfo, tasks, num_train=None, num_test=None, batch=False, batch_size=64, shuffle=True, num_workers=32):
    '''
    Dataloader for TIMIT.
    '''
    info_dict = build_info_dict(timitinfo)
    trnls = [l.rstrip('\n') for l in open(trainlist)][:num_train]
    tesls = [l.rstrip('\n') for l in open(testlist)][:num_test]
    print('To load {:d} train samples, {:d} test samples.'.format(len(trnls), len(tesls)))

    # Split data
    totls = trnls + tesls
    trnls_split = []
    tesls_split = []
    i = 0  # init outer pointer
    while i < len(totls):
        spkname = totls[i].split('/')[2]
        cnt = 0  # counter for each speaker
        j = i  # init lookup pointer
        while totls[j].split('/')[2] == spkname:
            cnt += 1
            j += 1
            if j >= len(totls):
                break
        cnt_l = math.floor(cnt * 0.8)
        k = i
        while k < i + cnt_l:
            trnls_split.append(totls[k])
            k += 1
        while k < i + cnt:
            tesls_split.append(totls[k])
            k += 1
        i = j  # fast forward outer pointer
    print('Splitted to {:d} train samples, {:d} test samples.'.format(len(trnls_split), len(tesls_split)))

    # Train
    train_feat = []
    train_label = []
    for l in tqdm(trnls_split, desc='load train', leave=True):
        spkname = l.split('/')[2][1:]
        for t in tasks:
            train_label.append(info_dict[spkname][t])
        train_feat.append(np.loadtxt(os.path.join(featdir, l), delimiter=','))

    # Test
    test_feat = []
    test_label = []
    for l in tqdm(tesls_split, desc='load test', leave=True):
        spkname = l.split('/')[2][1:]
        for t in tasks:
            test_label.append(info_dict[spkname][t])
        test_feat.append(np.loadtxt(os.path.join(featdir, l), delimiter=','))

    train_feat = np.asarray(train_feat, dtype=np.float64)
    train_label = np.asarray(train_label, dtype=np.float64).reshape((-1, len(tasks)))
    test_feat = np.asarray(test_feat, dtype=np.float64)
    test_label = np.asarray(test_label, dtype=np.float64).reshape((-1, len(tasks)))

    # Normalize
    # train_feat = (train_feat - train_feat.mean()) / (train_feat.std())
    # test_feat = (test_feat - test_feat.mean()) / (test_feat.std())

    # Batch data
    if batch:
        # Convert to torch tensor
        train_feat = torch.from_numpy(train_feat).float().view(-1, 1, 414, 450)
        train_label = torch.from_numpy(train_label).float()
        test_feat = torch.from_numpy(test_feat).float().view(-1, 1, 414, 450)
        test_label = torch.from_numpy(test_label).float()

        train_data = TensorDataset(train_feat, train_label)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_data = TensorDataset(test_feat, test_label)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return train_loader, test_loader

    else:
        return train_feat, train_label, test_feat, test_label
