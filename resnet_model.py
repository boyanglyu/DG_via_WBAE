
import numpy as np
import torchvision.models
import os
import sys
import torch
# import ssl

import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat

# ssl._create_default_https_context = ssl._create_unverified_context

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Clf(nn.Module):
    def __init__(self, in_features=200, out_features=65):
        super(Clf, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
    def forward(self, x):
        out = self.fc1(x)
        return out



class ResNet_MIGE(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, dropout):
        super(ResNet_MIGE, self).__init__()

        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        # save memory
        del self.network.fc
        # self.network.fc = Identity()
        self.network.fc = nn.Sequential(nn.Linear(2048, 1024), nn.Dropout(dropout),nn.ReLU(),
                                     nn.Linear(1024, 512),nn.Dropout(dropout), nn.ReLU(),
                                     nn.Linear(512, 200))
        self.freeze_bn()
        self.dropout = nn.Dropout(dropout)# this is a hyperparameter

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        out = self.dropout(self.network(x))
        return out

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()



class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self,dropout):
        super(ResNet, self).__init__()

        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        # save memory
        del self.network.fc
        # self.network.fc = Identity()
        self.network.fc = nn.Sequential(nn.Linear(2048, 1024), nn.Dropout(dropout),nn.ReLU(),
                                     nn.Linear(1024, 512),nn.Dropout(dropout), nn.ReLU(),
                                     nn.Linear(512, 200))

        self.freeze_bn()
        self.dropout = nn.Dropout(dropout) # this is a hyperparameter

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(200, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 2048)
        self.fc_bn2 = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU(inplace=True)
        # Decoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3) , stride= (2, 2),
                               padding=(0,0)),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5, 5), stride=(2, 2),
                               padding=(0,0)),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(5, 5), stride=(3, 3),
                               padding=(0, 0)),
            nn.BatchNorm2d(3, momentum=0.01),
        )
    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x))).view(-1, 32, 8, 8)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        # print(x.shape)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x
