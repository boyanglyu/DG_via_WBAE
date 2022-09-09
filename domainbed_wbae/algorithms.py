# Adapt from DomainBed code

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
############### BY added packaged ##################
from geomloss import SamplesLoss
import ot
############### BY added packaged ##################


ALGORITHMS = [
    'ERM',
    'WBAE'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



class WBAE(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(WBAE, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.decoder = networks.Decoder(input_shape)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters())+\
            list(self.decoder.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
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
        
        num_dist = len(latent_code)
        
        # when calculating the barycenter, disable the backpropagation
        with torch.no_grad():
            data_num, latent_dim = latent_code[0].shape
            for i in range(int(num_dist)):
                n_i = len(latent_code[i])
                b_i = ot.unif(n_i)
                x_i = latent_code[i].cpu().numpy()
                
                measures_locations.append(x_i)
                measures_weights.append(b_i)

            b = np.ones((num_dirac,)) / num_dirac
            X_init = np.random.normal(0., 1., (num_dirac, latent_dim))
            
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

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        alpha = self.hparams['wbae_alpha']
        beta = self.hparams['wbae_beta']
        blur = 20
        
        nmb = len(minibatches)
        
        input_img = [xi for xi, _ in minibatches]
        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]
        recons = [self.decoder(fi) for fi in features]
        
        clf_loss = 0
        recon_loss = 0
        objective = 0
        
        for i in range(nmb):
            clf_loss += F.cross_entropy(classifs[i], targets[i])
            recon_loss += F.mse_loss(recons[i], input_img[i]) * beta
        
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


