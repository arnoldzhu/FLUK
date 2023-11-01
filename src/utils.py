#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Lambda
from sampling import mnist_iid, fmnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sklearn.cluster import KMeans
import math
import UdacityDataset as UD
import data_utils

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users

        if args.iid:
            # Sample IID user data from Mnist
            # print('Num Users:' + args.num_users)
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                            transform=apply_transform,
                                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                            transform=apply_transform,
                                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        if args.iid:
            user_groups = fmnist_iid(train_dataset, args.num_users)

        else:
            raise NotImplementedError()

    elif args.dataset == 'udacity':
        train_dataset = UD.UdacityDataset(csv_file='',
                             root_dir=,
                            transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]),
                            select_camera='center_camera')
        
        test_dataset = UD.UdacityTestset(csv_file='',
                             root_dir=,
                            transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]),
                            select_camera='center_camera')
        
        user_groups = data_utils.udacity_split(train_dataset)

    else:
        exit('Unrecognized dataset!')

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Param w is a list of local weights after one round.
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def get_malicious_scores(kl_scores, num_clusters):
    kmeans = KMeans(
        init="random",
        n_clusters = num_clusters,
        n_init=10,
        random_state=42
    )
    kmeans.fit(kl_scores)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    label_count, major_label = 0, 0
    for label in kmeans.labels_:
        if label == 0:
            label_count += 1
    if label_count < len(kl_scores) / 2:
        major_label = 1
        
    malicious_scores = []
        
    for (hor, ver) in kl_scores:
        dist = math.sqrt((hor - kmeans.cluster_centers_[major_label][0]) ** 2 + (ver - kmeans.cluster_centers_[major_label][1]) ** 2)
        malicious_scores.append(dist)
        
    return None
