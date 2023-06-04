# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

#  import higher

from domainbed import networks
# from domainbed.lib.misc import random_pairs_of_minibatches, PJS_loss
from domainbed.optimizers import get_optimizer
from geomloss import SamplesLoss
import ot

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)



class WBAE(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(WBAE, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.decoder = networks.Decoder(input_shape)

        self.optimizer = get_optimizer(hparams["optimizer"],
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters())+\
            list(self.decoder.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
    '''
    latent_code: the extracted feature, list: [[tensor: # samples * latent_dim env1], [tensor: # samples * latent_dim env2], ...]
    num_dirac: number of support points for barycenter
    return barycenter X: shape is [num_dirac *  latent_dim]
    '''
    @staticmethod
    def wass_barycenter_new(latent_code, num_dirac=100):
        measures_locations = []
        measures_weights = []
        # get number of source domains in the latent code
        # num_dist = len(torch.unique(source_label))
        
        num_dist = len(latent_code)
        # print('number of sources is ', num_dist)
        # when calculating the barycenter, disable the backpropagation
        with torch.no_grad():
            data_num, latent_dim = latent_code[0].shape
            for i in range(int(num_dist)):
                n_i = len(latent_code[i])
                # print(n_i, ' elements from ', 'source ', i)
                b_i = ot.unif(n_i)
                # print('torch.where(one_source_idx)', torch.where(one_source_idx))
                x_i = latent_code[i].cpu().numpy()
                
                measures_locations.append(x_i)
                measures_weights.append(b_i)
            
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
    letent_code: list, lenth is number of training domian
    '''
    @staticmethod
    def wass_loss(latent_code, blur, num_dirac, device):
        # barycenter: shape is num_dirac *  latent_dim, numpy array
        barycenter = WBAE.wass_barycenter_new(latent_code, num_dirac)
        barycenter = torch.from_numpy(barycenter).type('torch.FloatTensor').to(device)
        num_dist = len(latent_code)
        total_loss = 0
        for i in range(num_dist):
            x_i = latent_code[i]
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True)
            total_loss += loss(x_i, barycenter.detach())
        return total_loss / num_dist

    def update(self,  x, y, **kwargs):
        device = "cuda" if x[0][0].is_cuda else "cpu"
        alpha = self.hparams['wbae_alpha']
        beta = self.hparams['wbae_beta']
        blur = 20
        nmb = len(x)
        
        features = [self.featurizer(xi) for xi in x]
        classifs = [self.classifier(fi) for fi in features]
        recons = [self.decoder(fi) for fi in features]
        
        clf_loss = 0
        recon_loss = 0
        objective = 0
        
        for i in range(nmb):
            clf_loss += F.cross_entropy(classifs[i], y[i])
            recon_loss += F.mse_loss(recons[i], x[i]) * beta
        
        clf_loss /= nmb
        recon_loss /= nmb
        wass_loss = WBAE.wass_loss(features, blur=blur, num_dirac=100, device=device) * alpha

        
        objective += clf_loss
        objective += wass_loss
        objective += recon_loss
        
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 
                'wass_loss': wass_loss.item(),
                'recon_loss': recon_loss.item(), 
                'clf_loss': clf_loss.item()}

    def predict(self, x):
        return self.network(x)

