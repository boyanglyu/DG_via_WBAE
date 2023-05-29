# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import math

from domainbed.lib import wide_resnet


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

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
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

class MCdropClassifier(nn.Module):
    def __init__(self, in_features, num_classes, bottleneck_dim=512, dropout_rate=0.5, dropout_type='Bernoulli'):
        super(MCdropClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.bottleneck_drop = self._make_dropout(dropout_rate, dropout_type)

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            self.bottleneck_drop
        )

        self.prediction_layer = nn.Linear(bottleneck_dim, num_classes)

    def _make_dropout(self, dropout_rate, dropout_type):
        if dropout_type == 'Bernoulli':
            return nn.Dropout(dropout_rate)
        elif dropout_type == 'Gaussian':
            return GaussianDropout(dropout_rate)
        else:
            raise ValueError(f'Dropout type not found')

    def activate_dropout(self):
        self.bottleneck_drop.train()

    def forward(self, x):
        hidden = self.bottleneck_layer(x)
        pred = self.prediction_layer(hidden)
        return pred


class GaussianDropout(nn.Module):
    def __init__(self, drop_rate):
        super(GaussianDropout, self).__init__()
        self.drop_rate = drop_rate
        self.mean = 1.0
        self.std = math.sqrt(drop_rate / (1.0 - drop_rate))

    def forward(self, x):
        if self.training:
            gaussian_noise = torch.randn_like(x, requires_grad=False).to(x.device) * self.std + self.mean
            return x * gaussian_noise
        else:
            return x


class Decoder_MNIST(nn.Module):
    n_outputs = 128

    # input_shape should be the shape of one element
    def __init__(self, input_shape):
        super(Decoder_MNIST, self).__init__()
        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(14, 14)),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_shape[0], 3, 1, padding=1) ### 64,3,3,1
        )
    def forward(self, x):
        x = x.view(-1,128,1,1)
        out = self.deConv(x)
        out = F.interpolate(out, size=(28, 28), mode='bilinear')
        return out

# Generator Code
class Decoder_Resnet(nn.Module):
    def __init__(self):
        super(Decoder_Resnet, self).__init__()

        # Decoder
        self.main = nn.Sequential(
            # first layer
            nn.ConvTranspose2d( 2048, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # second layer
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # forth layer 
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # fifth layer
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # sixth layer
            nn.ConvTranspose2d( 32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):

        # Decoder
        out = x.view(-1, 2048, 1, 1)
        out = self.main(out)
        # print(out.shape) (3, 128, 128)
        out = F.interpolate(out, size=(224, 224), mode='bilinear')
        return out

def Decoder(input_shape):
    """Auto-select an appropriate decoder for the given input shape."""
    if input_shape[1:3] == (28, 28):
        return Decoder_MNIST(input_shape)
    elif input_shape[1:3] == (224, 224):
        return Decoder_Resnet()
    else:
        raise NotImplementedError
