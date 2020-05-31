# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:57:22 2020

@author: Ashima
"""

import torch
import config
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Discriminator_Network(nn.Module):
    """Define Discriminator network of GAN."""

    def __init__(self, input_dim=config.IMAGE_SIZE,
                 output_dim=config.NUM_CLASSES):
        super(Discriminator_Network, self).__init__()
        self.input_dim = input_dim
        # Weight norm layers
        self.layers = torch.nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels=1, out_channels=32,
                                  kernel_size=3, stride=2, padding=1)),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=5, stride=1, padding=1)),
            weight_norm(nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=5, stride=1, padding=1))]
        )
        self.final = weight_norm(nn.Linear(128, config.NUM_FEATURES))
        self.n_layers = 3

    def forward(self, x, feature=False):
        """Forward computations of the discriminator network."""
        x = F.dropout(x, p=0.2)
        for i in range(self.n_layers):
            layer = self.layers[i]
            x = F.dropout(F.leaky_relu(layer(x), negative_slope=0.2), p=0.5)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        if feature:
            return x, x
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.final(x)


class Generator_Network(nn.Module):
    """Define Generator network of GAN."""

    def __init__(self, z_dim, output_dim=config.IMAGE_SIZE):
        super(Generator_Network, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(in_features=z_dim, out_features=10*10*128,
                             bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                          kernel_size=5, stride=1, padding=1,
                                          bias=False)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=5, stride=1, padding=1,
                                          bias=False)
        self.deconv3 = weight_norm(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1, kernel_size=4, stride=2,
                padding=1))

        self.bn1 = nn.BatchNorm1d(10*10*128, affine=False,
                                        eps=1e-6, momentum=0.5)
        self.bn2 = nn.BatchNorm2d(64)#, affine=False, eps=1e-6, momentum=0.5)
        self.bn3 = nn.BatchNorm2d(32)#, affine=False, eps=1e-6, momentum=0.5)
        self.bn1_b = Parameter(torch.zeros(10*10*128))

    def forward(self, batch_size):
        """Forward computations of the generator network."""
        x = Variable(torch.randn(batch_size, self.z_dim),
                     requires_grad=False,
                     volatile=not self.training).to(config.device)
        x = self.bn1(F.relu(self.fc1(x) + self.bn1_b))
        x = F.relu(self.bn2(self.deconv1(x.view(x.size(0), 128, 10, 10))))
        x = F.relu(self.bn3(self.deconv2(x)))
        x = F.sigmoid(self.deconv3(x))
        return x
