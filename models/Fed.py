#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg

def WAvg(w,weight):
    w_avg = copy.deepcopy(w[0])
    
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*weight[i]
        #w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg

# def FedAvg(w):
#     w_avg1 = copy.deepcopy(w[0])
#     w_avg2 = copy.deepcopy(w[5])
#     for k in w_avg1.keys():
#         for i in range(1, 5):
#             w_avg1[k] += w[i][k]
#             w_avg2[k] += w[i+5][k]
#         w_avg1[k] = torch.true_divide(w_avg1[k], 5)
#         w_avg2[k] = torch.true_divide(w_avg2[k], 5)
        
#     w_avg = copy.deepcopy(w_avg1)
#     for k in w_avg.keys():
#         w_avg[k] += w_avg2[k]
#         w_avg[k] = torch.true_divide(w_avg[k],2)
#     return w_avg 