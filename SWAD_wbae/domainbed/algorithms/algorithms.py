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


# Norm Version
class Independent_Inv(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Independent_Inv, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.featurizer_1 = networks.Featurizer(input_shape, self.hparams)
        self.featurizer_2 = networks.Featurizer(input_shape, self.hparams)
        
        self.classifier_1 = nn.Linear(self.featurizer_1.n_outputs, num_classes)
        self.classifier_2 = nn.Linear(self.featurizer_2.n_outputs, num_classes)

        self.classifier_all = nn.Linear(
            self.featurizer_1.n_outputs * 2,
            num_classes)
    
        self.optimizer_1 = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer_1.parameters()) +\
            list(self.featurizer_2.parameters())+\
            list(self.classifier_1.parameters())+\
            list(self.classifier_2.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # torch.optim.Adam(
        #     list(self.featurizer_1.parameters()) +\
        #     list(self.featurizer_2.parameters())+\
        #     list(self.classifier_1.parameters())+\
        #     list(self.classifier_2.parameters()),
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams['weight_decay']
        # )
        self.optimizer_2 = get_optimizer(
            hparams["optimizer"],
            self.classifier_all.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        device = "cuda" if x[0][0].is_cuda else "cpu"
        alpha = self.hparams['alpha']
        beta = self.hparams['beta']
        nmb = len(x)
        
        # First step
        features_1 = [self.featurizer_1(xi) for xi in x]
        classifs_1 = [self.classifier_1(fi) for fi in features_1]

        features_2 = [self.featurizer_2(xi) for xi in x]
        classifs_2 = [self.classifier_2(fi) for fi in features_2]
        
        clf_loss_1 = 0.
        clf_loss_2 = 0.
        
        penalty = 0.

        # Calculate CE loss and Coral Loss
        for i in range(nmb):
            clf_loss_1 += F.cross_entropy(classifs_1[i], y[i])
            clf_loss_2 += F.cross_entropy(classifs_2[i], y[i])
            for j in range(i + 1, nmb):
                penalty += self.coral(features_1[i], features_1[j])
                penalty += self.coral(features_2[i], features_2[j])
        clf_loss_1 /= nmb
        clf_loss_2 /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        # Calculate Disentanble loss
        
        z1, z2 = torch.cat(features_1), torch.cat(features_2)

        # Distangle by Covariance Matrix
        mean_z1 = torch.mean(z1, 0)
        mean_z2 = torch.mean(z2, 0)

        z1_n = z1 - mean_z1[None, :]
        z2_n = z2 - mean_z2[None, :]
        C = z1_n[:, :, None] * z2_n[:, None, :]

        target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).to(device)
        disentangle_loss = nn.MSELoss()(C, target_cr)
        
        total_loss = clf_loss_1 + clf_loss_2 + penalty * alpha + disentangle_loss * beta

        self.optimizer_1.zero_grad()
        total_loss.backward()
        self.optimizer_1.step()

        # Second step
        # input_img = torch.cat([xi for xi, _ in minibatches])
        all_y = torch.cat(y)
        with torch.no_grad():
            z1_norm = z1 / torch.linalg.norm(z1, dim=1, keepdim = True)
            z2_norm = z2 / torch.linalg.norm(z2, dim=1, keepdim = True)
            z = torch.cat((z1_norm.detach(), z2_norm.detach()), 1)
        predicted_classes = self.classifier_all(z)

        final_loss = F.cross_entropy(predicted_classes, all_y)

        self.optimizer_2.zero_grad()
        final_loss.backward()
        self.optimizer_2.step()

        return {'clf_loss_1': clf_loss_1.item(), 
                'clf_loss_2': clf_loss_2.item(), 
                'coral_loss': penalty.item(), 
                'cov_loss': disentangle_loss.item(),
                'total_loss': total_loss.item(),
                'final_loss': final_loss.item()}

    def predict(self, x):
        z1 = self.featurizer_1(x)
        z2 = self.featurizer_2(x)
        z1_norm = z1 / torch.linalg.norm(z1, dim=1, keepdim = True)
        z2_norm = z2 / torch.linalg.norm(z2, dim=1, keepdim = True)
        z = torch.cat((z1_norm, z2_norm), 1)
        return self.classifier_all(z)

class Independent_Inv_VRex(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Independent_Inv_VRex, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.featurizer_1 = networks.Featurizer(input_shape, self.hparams)
        self.featurizer_2 = networks.Featurizer(input_shape, self.hparams)
        
        self.classifier_1 = nn.Linear(self.featurizer_1.n_outputs, num_classes)
        self.classifier_2 = nn.Linear(self.featurizer_2.n_outputs, num_classes)

        self.classifier_all = nn.Linear(
            self.featurizer_1.n_outputs * 2,
            num_classes)
    
        self.optimizer_1 = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer_1.parameters()) +\
            list(self.featurizer_2.parameters())+\
            list(self.classifier_1.parameters())+\
            list(self.classifier_2.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # torch.optim.Adam(
        #     list(self.featurizer_1.parameters()) +\
        #     list(self.featurizer_2.parameters())+\
        #     list(self.classifier_1.parameters())+\
        #     list(self.classifier_2.parameters()),
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams['weight_decay']
        # )
        self.optimizer_2 = get_optimizer(
            hparams["optimizer"],
            self.classifier_all.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update_classifier_all(self, z1, z2, targets):
        '''
        z1, z2:  concatenated features with the same dimension, normalized to make them have unit length
        target: a list of minibatch labels
        We referenced the Vrex method, the classifier_all is trained to minimized 
        both the mean and the variance of two independent loss
        '''
        device = "cuda" if targets[0][0].is_cuda else "cpu"
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        y = torch.cat(targets)
        with torch.no_grad():
            # filled_zero = torch.zeros(z1.size()).to(device)
            z1_full = torch.cat((z1, z1), 1)
            # z2_full = torch.cat((filled_zero, z2), 1) Thuan TODO
            z2_full = torch.cat((z2, z2), 1) 
            z_all = torch.cat((z1, z2), 1)
        
        z1_logits = self.classifier_all(z1_full.detach())
        z2_logits = self.classifier_all(z2_full.detach())
        z_all_logits = self.classifier_all(z_all.detach())

        losses = torch.zeros(3).to(device)
        
        losses[0] = F.cross_entropy(z1_logits, y)
        losses[1] = F.cross_entropy(z2_logits, y)
        losses[2] = F.cross_entropy(z_all_logits, y)

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer_2 = torch.optim.Adam(
                self.classifier_all.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer_2.zero_grad()
        loss.backward()
        self.optimizer_2.step()

        self.update_count += 1
        return loss.item(), mean.item(), penalty.item()

    def update(self, x, y, **kwargs):
        device = "cuda" if x[0][0].is_cuda else "cpu"
        alpha = self.hparams['alpha']
        beta = self.hparams['beta']
        nmb = len(x)
        
        # First step
        features_1 = [self.featurizer_1(xi) for xi in x]
        classifs_1 = [self.classifier_1(fi) for fi in features_1]

        features_2 = [self.featurizer_2(xi) for xi in x]
        classifs_2 = [self.classifier_2(fi) for fi in features_2]
        
        clf_loss_1 = 0.
        clf_loss_2 = 0.
        
        penalty = 0.

        # Calculate CE loss and Coral Loss
        for i in range(nmb):
            clf_loss_1 += F.cross_entropy(classifs_1[i], y[i])
            clf_loss_2 += F.cross_entropy(classifs_2[i], y[i])
            for j in range(i + 1, nmb):
                penalty += self.coral(features_1[i], features_1[j])
                penalty += self.coral(features_2[i], features_2[j])
        clf_loss_1 /= nmb
        clf_loss_2 /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        # Calculate Disentanble loss
        
        z1, z2 = torch.cat(features_1), torch.cat(features_2)

        # Distangle by Covariance Matrix
        mean_z1 = torch.mean(z1, 0)
        mean_z2 = torch.mean(z2, 0)

        z1_n = z1 - mean_z1[None, :]
        z2_n = z2 - mean_z2[None, :]
        C = z1_n[:, :, None] * z2_n[:, None, :]

        target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).to(device)
        disentangle_loss = nn.MSELoss()(C, target_cr)
        
        total_loss = clf_loss_1 + clf_loss_2 + penalty * alpha + disentangle_loss * beta

        self.optimizer_1.zero_grad()
        total_loss.backward()
        self.optimizer_1.step()

        
        # Second step
        with torch.no_grad():
            z1_norm = z1 / torch.linalg.norm(z1, dim=1, keepdim = True)
            z2_norm = z2 / torch.linalg.norm(z2, dim=1, keepdim = True)
        final_loss, mean, variance =  self.update_classifier_all(z1_norm, z2_norm, y)
       
        # all_y = torch.cat(y)
        # with torch.no_grad():
        #     z1_norm = z1 / torch.linalg.norm(z1, dim=1, keepdim = True)
        #     z2_norm = z2 / torch.linalg.norm(z2, dim=1, keepdim = True)
        #     z = torch.cat((z1_norm.detach(), z2_norm.detach()), 1)
        # predicted_classes = self.classifier_all(z)

        # final_loss = F.cross_entropy(predicted_classes, all_y)

        # self.optimizer_2.zero_grad()
        # final_loss.backward()
        # self.optimizer_2.step()

        return {'clf_loss_1': clf_loss_1.item(), 
                'clf_loss_2': clf_loss_2.item(), 
                'coral_loss': penalty.item(), 
                'cov_loss': disentangle_loss.item(),
                'final_loss': final_loss,
                'mean': mean,
                'variance': variance}
    def predict(self, x):
        z1 = self.featurizer_1(x)
        z2 = self.featurizer_2(x)
        z1_norm = z1 / torch.linalg.norm(z1, dim=1, keepdim = True)
        z2_norm = z2 / torch.linalg.norm(z2, dim=1, keepdim = True)
        z = torch.cat((z1_norm, z2_norm), 1)
        return self.classifier_all(z)

class DNA(Algorithm):
    """
    Diversified Neural Averaging(DNA)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DNA, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.MCdropClassifier(
            in_features=self.featurizer.n_outputs,
            num_classes=num_classes,
            bottleneck_dim=self.hparams["bottleneck_dim"],
            dropout_rate=self.hparams["dropout_rate"],
            dropout_type=self.hparams["dropout_type"]
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.train_sample_num = 5
        self.lambda_v = self.hparams["lambda_v"]

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_f = self.featurizer(all_x)
        loss_pjs = 0.0
        row_index = torch.arange(0, all_x.size(0))

        probs_y = []
        for i in range(self.train_sample_num):
            pred = self.classifier(all_f)
            prob = F.softmax(pred, dim=1)
            prob_y = prob[row_index, all_y]
            probs_y.append(prob_y.unsqueeze(0))
            loss_pjs += PJS_loss(prob, all_y)

        probs_y = torch.cat(probs_y, dim=0)
        X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log(2*probs_y/(1+probs_y)) + 1e-6)
        loss_v = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
        loss_pjs /= self.train_sample_num
        loss = loss_pjs - self.lambda_v * loss_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "loss_c": loss_pjs.item(), "loss_v": loss_v.item()}

    def predict(self, x):
        return self.network(x)


