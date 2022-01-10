#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=768, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.02, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='vgg', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # group arguements
    parser.add_argument('--num_groups', type=int, default=5, help="number of groups")
    parser.add_argument('--interval', type=int, default=1, help="number of rounds before computing loss")
    parser.add_argument('--local_period', type=int, default=10,help='length of local period I')
    parser.add_argument('--group_freq', type=int, default=1, help='number of group period before global aggregation, G/I')

    parser.add_argument('--num_workers', type=int, default=0, help="number of workers when loading data")
    parser.add_argument('--team_epochs', type=int, default=1, help="number of team epochs")
    parser.add_argument('--num_teams', type=int, default=2, help="number of teams")

    args = parser.parse_args()
    return args
