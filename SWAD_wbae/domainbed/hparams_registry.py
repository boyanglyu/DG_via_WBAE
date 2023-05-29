# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}
    # (0, 5e-5, 1e-6), (0.1, 5e-5, 1e-6), (0.5, 5e-5, 1e-6), (0, 5e-5, 1e-4), (0.1, 5e-5, 1e-4), (0.5, 5e-5, 1e-4)
    # (0, 3e-5, 1e-6), (0.1, 3e-5, 1e-6), (0.5, 3e-5, 1e-6), (0, 3e-5, 1e-4), (0.1, 3e-5, 1e-4), (0.5, 3e-5, 1e-4)
    # (0, 1e-5, 1e-6), (0.1, 1e-5, 1e-6), (0.5, 1e-5, 1e-6), (0, 1e-5, 1e-4), (0.1, 1e-5, 1e-4), (0.5, 1e-5, 1e-4)
    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5])) # Here (0.5, 1e-5, 1e-4)
    hparams["class_balanced"] = (False, False)
    hparams["scheduler"] = ("const", "const")
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, random_state.choice([1e-5, 3e-5, 5e-5])) # Here
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (1e-6, random_state.choice([1e-6,1e-4])) # Here

    
    if algorithm == "WBAE":
        hparams['wbae_alpha'] = (10**(-3), random_state.choice([10**(-3.5), 10**(-3), 10**(-2.5), 10**(-2)]))  
        hparams['wbae_beta'] = (10**(-2), random_state.choice([10**(-3.5), 10**(-3), 10**(-2), 10**(-1.5)]))
        
 

    return hparams



def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
