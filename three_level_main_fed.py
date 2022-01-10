#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import time
import datetime
import torchvision.models as models
# from resnet import resnet20

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNEMnist, CNNCifar, ResNet
from models.Fed import FedAvg, WAvg
from models.test import test_img

from data_reader import femnist, celeba

if __name__ == '__main__':
    # parse args
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = 'cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'

    # load dataset and split users
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_path = os.path.join(os.path.dirname(__file__), 'dataset_files', 'femnist')
        dataset_train = femnist.FEMNIST(data_path, train=True, download=True, transform=trans_mnist)
        dataset_test = femnist.FEMNIST(data_path, train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = dataset_train.get_dict_clients()
            print('** resetting number of users according to the actual value in the dataset for FEMNIST non-IID **')
            args.num_users = len(dict_users.keys())
            print('number of users:', args.num_users)
    elif args.dataset == 'celeba':
        trans_celeba = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        data_path = os.path.join(os.path.dirname(__file__), 'dataset_files', 'celeba')
        dataset_train = celeba.CelebA(data_path, train=True, download=True, transform=trans_celeba)
        dataset_test = celeba.CelebA(data_path, train=False, download=True, transform=trans_celeba)
        # sample users
        if args.iid:
            raise Exception('iid case not implemented')
        else:
            dict_users = dataset_train.get_dict_clients()
            print('** resetting number of users according to the actual value in the dataset for CelebA non-IID **')
            args.num_users = len(dict_users.keys())
            print('number of users:', args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    stats = np.zeros(len(dict_users))
    for kk in range(0,len(dict_users)):
        stats[kk] = len(dict_users[kk])
    print(np.mean(stats))
    print(np.std(stats))
    print(len(dataset_train))
    print(dataset_test)




    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_glob = nn.DataParallel(net_glob)
    elif args.model == 'vgg':
        net_glob = models.vgg11(pretrained=False).to(args.device)
        net_glob = nn.DataParallel(net_glob)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = CNNEMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)
        net_glob = nn.DataParallel(net_glob)
        net_glob = net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()  # net initialization

    # copy weights
    w_glob = net_glob.state_dict()

    # initialization
    loss_train = []
    acc_train = []
    test_acc_train = []
    test_loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # group params
    num_samples = np.array([len(dict_users[i]) for i in dict_users])
    num_clients = int(args.num_users / args.num_groups)
    num_sampled = int(np.round(args.num_users * args.frac))
    group = []
    gweight = []
    num_sampled_clients = np.ones(args.num_groups, dtype=int) * int(num_clients * args.frac)
    num_sampled_clients[args.num_groups - 1] = num_sampled - np.sum(num_sampled_clients[0:args.num_groups - 1])
    # lweight = []
    for i in range(0, args.num_groups):
        if i == args.num_groups - 1:
            group.append(np.arange(i * num_clients, args.num_users))
            gweight.append(np.sum(num_samples[i * num_clients:args.num_users]) / np.sum(num_samples))
            # lweight.append(num_samples[i*num_clients:args.num_users]/np.sum(num_samples[i*num_clients:(i+1)*num_clients]))
        else:
            group.append(np.arange(i * num_clients, (i + 1) * num_clients))
            gweight.append(np.sum(num_samples[i * num_clients:(i + 1) * num_clients]) / np.sum(num_samples))
            # lweight.append(num_samples[i*num_clients:(i+1)*num_clients]/np.sum(num_samples[i*num_clients:(i+1)*num_clients]))
    # different I for different groups can be set here.
    group_epochs = np.ones(args.num_groups, dtype=int) * args.group_freq
    local_iters = np.ones(args.num_groups, dtype=int) * args.local_period
    # the number of local iterations can not exceed the number of local batches.

    # create a new level called team
    # args.num_teams
    team = []
    num_group = int(args.num_groups / args.num_teams)
    tweight = []
    gintweight = []
    for i in range(0, args.num_teams):
        if i == args.num_teams- 1:
            team.append(np.arange(i * num_group, args.num_groups))
            #gweight.append(np.sum(num_samples[i * num_clients:args.num_users]) / np.sum(num_samples))
            tweight.append(sum(gweight[i * num_group: args.num_groups]))
            gintweight.append(gweight[i * num_group: args.num_groups]/sum(gweight[i * num_group: args.num_groups]))
            # lweight.append(num_samples[i*num_clients:args.num_users]/np.sum(num_samples[i*num_clients:(i+1)*num_clients]))
        else:
            team.append(np.arange(i * num_group, (i + 1) * num_group))
            tweight.append(sum(gweight[i * num_group: (i + 1) * num_group]))
            gintweight.append(gweight[i * num_group: (i + 1) * num_group] / sum(gweight[i * num_group: (i + 1) * num_group]))
            #gweight.append(np.sum(num_samples[i * num_clients:(i + 1) * num_clients]) / np.sum(num_samples))
    team_epochs = args.team_epochs


    # training

    # filename
    f_l1 = args.dataset + '-3level-' + str(args.num_groups) + '-' + str(args.team_epochs) + '-' + str(args.group_freq) + '-' + str(
        args.local_period) + '-loss.txt'
    f_a1 = args.dataset + '-3level-' + str(args.num_groups) + '-' + str(args.team_epochs) + '-' + str(args.group_freq) + '-' + str(
        args.local_period) + '-acc.txt'
    f_l2 = args.dataset + '-3level-' + str(args.num_groups) + '-' + str(args.team_epochs) + '-' + str(args.group_freq) + '-' + str(
        args.local_period) + '-test_loss.txt'
    f_a2 = args.dataset + '-3level-' + str(args.num_groups) + '-' + str(args.team_epochs) + '-' + str(args.group_freq) + '-' + str(
        args.local_period) + '-test_acc.txt'

    stime = time.time()

    for iters in range(args.epochs):

        # print('******** global epoch', iters)
        # w_groups,loss_groups,acc_groups = [], [], []
        w_glob = None
        for team_idx in range(0, num_teams):
            net_team = copy.deepcopy(net_glob).to(args.device)
            for ii in range(0, team_epochs):
                w_team = None
                for group_idx in team[team_idx]:
                    # print('***** group_idx', group_idx)
                    net_group = copy.deepcopy(net_team).to(args.device)
                    loss_locals, acc_locals = [], []
                    # num_sampled = int(len(group[group_idx])*args.frac)
                    for i in range(0, group_epochs[group_idx]):
                        # print('***** i', i)

                        # w_locals = []
                        w_group = None

                        sampled_clients = np.array([])
                        sampled_clients = np.random.choice(group[group_idx], size=num_sampled_clients[group_idx], replace=False)

                        lweight = num_samples[sampled_clients]
                        lweight = lweight / np.sum(lweight)

                        for j in range(0, len(sampled_clients)):
                            idx = sampled_clients[j]
                            # print('***** idx', idx)

                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                                iters=local_iters[group_idx], nums=num_samples[idx])
                            net_group_local = copy.deepcopy(net_group).to(args.device)
                            w = local.train(net=net_group_local)
                            for k in w.keys():
                                w[k] = w[k].detach()

                            if w_group is None:
                                w_group = w
                                for k in w.keys():
                                    w_group[k] *= lweight[j]
                            else:
                                for k in w.keys():
                                    w_group[k] += w[k] * lweight[j]

                            del net_group_local

                            # w_locals.append(copy.deepcopy(w))
                            # if torch.cuda.is_available():
                            #    print('GPU memory allocated:', torch.cuda.memory_allocated(args.device))
                            #    print('GPU memory reserved:', torch.cuda.memory_reserved(args.device))

                        # group aggregation
                        # lweight = num_samples[sampled_clients]
                        # lweight = lweight/np.sum(lweight)
                        # with torch.no_grad():
                        #     w_group = WAvg(w_locals,lweight)
                        # w_group = FedAvg(w_locals)
                        net_group.load_state_dict(w_group)

            # w_groups.append(w_group)
                    if w_team is None:
                        w_team = w_group
                        for k in w_group.keys():
                            w_team[k] *= gintweight[group_idx]
                    else:
                        for k in w_group.keys():
                            w_team[k] += w_group[k] * gintweight[group_idx]
                net_team.load_state_dict(w_team)

        # global aggregation
        # with torch.no_grad():
        #     w_glob = WAvg(w_groups,gweight)
        # w_glob = FedAvg(w_groups)

        # copy weight to net_glob
            if w_glob is None:
                w_glob = w_team
                for k in w_team.keys():
                    w_glob[k] *= tweight[team_idx]
            else:
                for k in w_team.keys():
                    w_glob[k] += w_team[k] * tweight[team_idx]
        net_glob.load_state_dict(w_glob)

        # compute training/test accuracy/loss
        if (iters + 1) % args.interval == 0:
            with torch.no_grad():
                acc_avg, loss_avg = test_img(copy.deepcopy(net_glob).to(args.device), dataset_train, args)
                acc_test, loss_test = test_img(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
            print('Round {:3d}, Training loss {:.3f}'.format(iters, loss_avg), flush=True)
            print('Round {:3d}, Training acc {:.3f}'.format(iters, acc_avg), flush=True)
            print('Round {:3d}, Test loss {:.3f}'.format(iters, loss_test), flush=True)
            print('Round {:3d}, Test acc {:.3f}'.format(iters, acc_test), flush=True)
            # loss_train.append(loss_avg)
            # acc_train.append(acc_avg)
            # test_loss_train.append(loss_test)
            # test_acc_train.append(acc_test)

            # write into files
            with open(f_l1, 'a') as l1, open(f_a1, 'a') as a1, open(f_l2, 'a') as l2, open(f_a2, 'a') as a2:
                l1.write(str(loss_avg))
                l1.write('\n')
                a1.write(str(acc_avg))
                a1.write('\n')
                l2.write(str(loss_test))
                l2.write('\n')
                a2.write(str(acc_test))
                a2.write('\n')

    ftime = time.time() - stime
    ftime = datetime.timedelta(seconds=ftime)
    print("Training time {}".format(ftime))
    # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-loss.txt',loss_train)
    # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-acc.txt',acc_train)
    # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-test_loss',test_loss_train)
    # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-test_acc',test_acc_train)