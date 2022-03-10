#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:05:33 2020

@author: boyanglyu
"""
import ot
import numpy as np
import torch
from geomloss import SamplesLoss
from scipy.io import loadmat

np.random.seed(0)
torch.manual_seed(1)



'''
return number of correct predictions
'''
def accuracy(y_pred, y_test):
    _, predicted = torch.max(y_pred, 1)
    y_true = y_test
    correct = (predicted == y_true).sum().item()
    return correct


def split_dataset(_in, fraction):
    '''
    _in: (data,label) numpy array
    fraction: 0-1, proportation for training data
    return: train, validation data
    '''
    x, y = _in
    assert(fraction <= 1)
    n = int(fraction * len(y))
    keys = list(range(len(y)))
    # print(keys)
    np.random.shuffle(keys)

    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return x[keys_1], y[keys_1], x[keys_2], y[keys_2]



'''
Calculate the support points for the wasserstein barycenter of all source domains

latent_code: the extracted feature, shape [# samples * latent_dim], type tensor
source_label: the corresponding domain index the latent code belonging to,
              shape [#samples], type tensor
num_dirac: number of support points for barycenter
return barycenter X: shape is [num_dirac *  latent_dim]
'''
def wass_barycenter(latent_code, source_label, num_dirac=50):
    measures_locations = []
    measures_weights = []
    # get number of source domains in the latent code
    # num_dist = len(torch.unique(source_label))
    unique_slb = torch.unique(source_label)
    num_dist = len(unique_slb)
    # print('number of sources is ', num_dist)
    # when calculating the barycenter, disable the backpropagation
    with torch.no_grad():
        data_num, latent_dim = latent_code.shape
        for i in range(int(num_dist)):
            one_source_idx = source_label.eq(unique_slb[i])
            # print('one_source_idx',one_source_idx)
            n_i = one_source_idx.sum()
            # print(n_i, ' elements from ', 'source ', i)
            b_i = ot.unif(n_i.item())
            # print('torch.where(one_source_idx)', torch.where(one_source_idx))
            x_i = latent_code[torch.where(one_source_idx)]
            x_i = x_i.cpu().numpy()
            #
            measures_locations.append(x_i)
            measures_weights.append(b_i)
        # print(measures_locations, measures_weights)
        # weights of the barycenter
        b = np.ones((num_dirac,)) / num_dirac
        # number of Diracs of the barycenter
        X_init = np.random.normal(0., 1., (num_dirac, latent_dim))

        #measures_locations (list of N (k_i,d) numpy.ndarray) – The discrete
        #        support of a measure supported on k_i locations of a d-dimensional
        #        space (k_i can be different for each element of the list)
        # measures_weights (list of N (k_i,) numpy.ndarray) – Numpy arrays where
        #        each numpy array has k_i non-negatives values summing to one
        #        representing the weights of each discrete input measure
        # X_init ((k,d) np.ndarray) – Initialization of the support locations (on k atoms) of the barycenter
        # b ((k,) np.ndarray) – Initialization of the weights of the barycenter (non-negatives, sum to 1)
        X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init, b)
        return X

'''
latent_code: the extracted feature z
source_label: the corresponding domain the latent code belonging to
num_dirac: number of points for barycenter
blur: https://www.kernel-operations.io/geomloss/api/pytorch-api.html
num_dirac: number of support points for barycenter
return summation of wasserstein loss for distribution from different source domains
'''

def wass_loss(latent_code, source_label, blur, num_dirac=100, device='cpu'):
    # barycenter: shape is [num_dirac *  latent_dim]
    barycenter = wass_barycenter(latent_code, source_label, num_dirac)
    barycenter = torch.from_numpy(barycenter).type('torch.FloatTensor').to(device)

    unique_slb = torch.unique(source_label)
    num_dist = len(unique_slb)
    total_loss = 0
    for i in range(int(num_dist)):
        one_source_idx = source_label.eq(unique_slb[i])
        x_i = latent_code[torch.where(one_source_idx)]

        # https://www.kernel-operations.io/geomloss/api/pytorch-api.html
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True)
        total_loss += loss(x_i, barycenter.detach())
    return total_loss #/ int(num_dist)
