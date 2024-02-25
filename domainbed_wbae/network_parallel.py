# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy
from torch.distributed.pipeline.sync import Pipe


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class PiplineParallelResNet(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(PiplineParallelResNet, self).__init__()
        self.temp = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048
        nc = input_shape[0]
        del self.temp.fc
        self.temp.fc = Identity()
        self.num_gpu = torch.cuda.device_count()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.network = self.split_model()

    def split_model(self):
        children = list(self.temp.children())
        split_size = len(children) // self.num_gpu
        segments = []
      
        for i in range(0, self.num_gpu):
            start_idx = i * split_size
            # For the last GPU, include all remaining layers
            end_idx = (i + 1) * split_size if i < self.num_gpu - 1 else len(children)
            segment = nn.Sequential(*children[start_idx:end_idx]).cuda(i)
            segments.append(segment)

        return segments
    
    def forward(self, x):
        for i, segment in enumerate(self.network):
            x = segment(x.to(f'cuda:{i}'))
        return self.dropout(x).squeeze()

def Featurizer(input_shape, hparams):
    pipline_segments = PiplineParallelResNet(input_shape, hparams).network
    return Pipe(torch.nn.Sequential(*pipline_segments), chunks=8)

def Classifier(in_features, out_features, is_nonlinear=False):
    last_gpu = torch.cuda.device_count() - 1
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features)).to(f'cuda:{last_gpu}')
    else:
        return torch.nn.Linear(in_features, out_features).to(f'cuda:{last_gpu}')


