#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



#group-IID realization
def cifar_giid(dataset,num_users):
    labels = np.arange(0,len(dataset))
    idxs = np.arange(0,len(dataset))
    for i in range(0,len(dataset)):
        labels[i] = dataset[i][1]
    dict_users = {i: np.array([], dtype='int32') for i in range(num_users)}
    group1_idx = np.random.choice(idxs,25000,replace=False)
    group2_idx = np.array(list(set(idxs)-set(group1_idx)))
    
    
    for i in range(0,5):
        dict_users[i] = group1_idx[labels[group1_idx]==i]
        dict_users[i] = np.concatenate((dict_users[i],group1_idx[labels[group1_idx]==(i+5)]), axis=0)
        dict_users[i+5] = group2_idx[labels[group2_idx]==i]
        dict_users[i+5] = np.concatenate((dict_users[i+5],group2_idx[labels[group2_idx]==(i+5)]), axis=0)

    return dict_users





def cifar_iid(dataset,num_users):
    l = len(dataset)
    idxs = np.arange(0,l)
    np.random.shuffle(idxs)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    shard = int(l/num_users)
    for i in range(0,num_users):
        dict_users[i] = idxs[i*shard:(i+1)*shard]
    return dict_users




def cifar_noniid(dataset,num_users):
    l = len(dataset)
    labels = np.arange(0,l)
    idxs = np.arange(0,l)
    for i in range(0,l):
        labels[i] = dataset[i][1]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    shard = int(l/num_users)

    for i in range(0,num_users):
        dict_users[i] = idxs[i*shard:(i+1)*shard]
        
    return dict_users












if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
