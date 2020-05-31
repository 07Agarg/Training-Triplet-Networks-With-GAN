# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:21:47 2020

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

    def __init__(self, input_dim=28 ** 2, output_dim=config.NUM_CLASSES):
        super(Discriminator_Network, self).__init__()
        self.input_dim = input_dim
        # Weight norm layers
        self.layers = torch.nn.ModuleList([
            weight_norm(nn.Linear(input_dim, 1000)),
            weight_norm(nn.Linear(1000, 500)),
            weight_norm(nn.Linear(500, 250)),
            weight_norm(nn.Linear(250, 250)),
            weight_norm(nn.Linear(250, 250))]
        )
        self.final = weight_norm(nn.Linear(250, config.NUM_FEATURES))
        self.n_layers = 5

    def _add_noise(self, x, sigma):
        """Sample noise of given variance."""
        if self.training:
            noise = torch.randn(x.size()).to(config.device) * sigma
        else:
            noise = torch.Tensor([0]).to(config.device)
        noise = Variable(noise, requires_grad=False)
        return noise

    def forward(self, x, feature=False):
        """Forward computations of the discriminator network."""
        x = x.view(-1, self.input_dim)
        x = x + self._add_noise(x, sigma=0.3)
        for i in range(self.n_layers):
            layer = self.layers[i]
            x_f = F.relu(
                layer(x))
            x = x_f + self._add_noise(x_f, sigma=0.5)
        if feature:
            return x_f, self.final(x)
        return self.final(x)


class Generator_Network(nn.Module):
    """Define Generator network of GAN."""

    def __init__(self, z_dim, output_dim=config.IMAGE_SIZE):
        super(Generator_Network, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 250, bias=False)
        self.fc2 = nn.Linear(250, 250, bias=False)
        self.fc3 = nn.Linear(250, 500, bias=False)
        self.fc4 = nn.Linear(500, 1000, bias=False)
        self.fc5 = weight_norm(nn.Linear(1000, output_dim, bias=False))
        self.bn1 = nn.BatchNorm1d(250, affine=False, eps=1e-6, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(250, affine=False, eps=1e-6, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(500, affine=False, eps=1e-6, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(1000, affine=False, eps=1e-6, momentum=0.5)
        
        self.bn1_b = Parameter(torch.zeros(250))
        self.bn2_b = Parameter(torch.zeros(250))
        self.bn3_b = Parameter(torch.zeros(500))
        self.bn4_b = Parameter(torch.zeros(1000))
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)
        
    def forward(self, batch_size):
        """Forward computations of the generator network."""
        x = Variable(torch.rand(batch_size, self.z_dim),
                     requires_grad=False,
                     volatile=not self.training).to(config.device)
        x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.bn3(self.fc3(x)) + self.bn3_b)
        x = F.softplus(self.bn4(self.fc4(x)) + self.bn4_b)
        x = F.sigmoid(self.fc5(x))
        return x
