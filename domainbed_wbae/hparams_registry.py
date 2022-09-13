# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: 0.) #   fixed the dropout
    # _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

   
    if algorithm == "WBAE":
        _hparam('wbae_alpha', 1e-3, lambda r: 10**r.uniform(-3.5, -2))
        _hparam('wbae_beta', 1e-2, lambda r: 10**r.uniform(-3.5, -1.5)) 
        _hparam('wbae_blur', 20., lambda r: 20.) #0729

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 1e-3) #   fixed the learning rate 
    else:
         _hparam('lr', 5e-5, lambda r: 5e-5) #   fixed the learning rate 

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)#   fixed the learning rate 
    else:
        _hparam('weight_decay', 0., lambda r: 0.) #   fixed the weight decay

    if dataset in SMALL_IMAGES:
        # _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
        _hparam('batch_size', 64, lambda r: 64) #   fixed the batch size
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: 32) #   fixed the batch size
        
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
