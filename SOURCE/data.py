# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:29:08 2020

@author: Ashima
"""

import time
import torch
import utils
import config
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import LabelEncoder


class TripletDataset(Dataset):
    """Load MNIST dataset and sample labeled dataset of
        equal samples from each class."""

    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])         #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = MNIST(root=config.DATA_DIR, train=True,
                             transform=transform, download=True)
        class_tot = [0] * config.NUM_CLASSES
        data = []
        labels = []
        tot = 0
        perm = np.random.permutation(self.dataset.__len__())
        for i in range(self.dataset.__len__()):
            datum, label = self.dataset.__getitem__(perm[i])
            if class_tot[label] < config.NUM_LABELED_SAMPLES_PER_CLASS:
                data.append(datum.numpy())
                labels.append(label)
                class_tot[label] += 1
                tot += 1
                if tot >= 10 * config.NUM_LABELED_SAMPLES_PER_CLASS:
                    break
        self.data = data
        self.labels = np.array(labels)
        self.make_triplet_list(config.NUM_TRIPLETS)

    def make_triplet_list(self, ntriplets):
        """Create triplet list."""
        print('Processing Triplet Generation ...')
        triplets = []
        for class_idx in range(config.NUM_CLASSES):
            start_time = time.time()
            a = np.random.choice(np.where(self.labels==class_idx)[0],
                                 int(ntriplets/config.NUM_CLASSES),
                                 replace=True)
            b = np.random.choice(np.where(self.labels==class_idx)[0],
                                 int(ntriplets/config.NUM_CLASSES),
                                 replace=True)
            # while np.any((a-b)==0):
            #     np.random.shuffle(b)
            c = np.random.choice(np.where(self.labels != class_idx)[0],
                                 int(ntriplets/10), replace=True)
            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(b[i]), int(c[i])])
            print("Time for class idx {} is {}".format(class_idx,
                                                (time.time() - start_time)))
        self.triplets = triplets

    def __len__(self):
        """Return the total no of triplets."""
        return len(self.triplets)

    def __getitem__(self, index):
        """Return a triplet."""
        index1, index2, index3 = self.triplets[index]
        img1, img2, img3 = self.data[index1], self.data[index2], self.data[index3]
        return img1, img2, img3


class Data():
    def __init__(self):
        self.dataset = None
        self.train_dataloader = None
        self.size = 0

    def read(self, train=True):
        """Read daataset and create dataloaders for train and test data."""
        trans = transforms.Compose([transforms.ToTensor()])         #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if train:
            self.unlabeled_dataset = MNIST(root=config.DATA_DIR, train=train,
                                           transform=trans, download=True)
            self.labeled_dataset = TripletDataset()
            self.labeled_dataloader = DataLoader(self.labeled_dataset,
                                                  config.BATCH_SIZE, shuffle=True,
                                                  drop_last=True)
            self.unlabeled_dataloader_disc = DataLoader(self.unlabeled_dataset,
                                                   config.BATCH_SIZE,
                                                   shuffle=True,
                                                   drop_last=True)
            self.unlabeled_dataloader_gen = DataLoader(self.unlabeled_dataset,
                                                   config.BATCH_SIZE,
                                                   shuffle=True,
                                                   drop_last=True)
            self.labeled_iterator = iter(self.labeled_dataloader)
            self.unlabeled_iterator_disc = iter(self.unlabeled_dataloader_disc)
            self.unlabeled_iterator_gen = iter(self.unlabeled_dataloader_gen)
            self.dataset = self.unlabeled_dataset
        else:
            self.test_dataset = MNIST(root=config.DATA_DIR, train=train,
                                         transform=trans, download=True)
            self.test_dataloader = DataLoader(self.test_dataset,
                                              config.BATCH_SIZE,
                                              shuffle=True,
                                              drop_last=True)
            self.test_iterator = iter(self.test_dataloader)
            self.dataset = self.test_dataset

    def load_unsupervised_batch(self, dataloader, iterator):
        """Load a batch of data."""
        if len(dataloader) == 0:
            return [], []
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            x, y = next(iterator)
        return x.float().to(config.device), y.long().to(config.device)

    def load_triplet_batch(self, dataloader, iterator):
        """Load a triplet (x_anchor, x_pos, x_neg).
        x_anchor and x_pos belongs to same class and x_neg belongs
        to class different from that of x_anchor."""
        if len(dataloader) == 0:
            return [], [], []
        try:
            x_anchor, x_pos, x_neg = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            x_anchor, x_pos, x_neg = next(iterator)
        return [x_anchor.float().to(config.device),
                x_pos.float().to(config.device),
                x_neg.float().to(config.device)]

    def load_features(self, net, string="train"):
        """Extract features from the trained Discriminator model."""
        net.load_state_dict(torch.load(utils.discriminator_path()))
        net.eval()
        embeddings_list = []
        labels_list = []

        if string == "train":
            dataloader = self.unlabeled_dataloader_disc
            iterator = self.unlabeled_iterator_disc
        elif string == "test":
            dataloader = self.test_dataloader
            iterator = self.test_iterator

        with torch.no_grad():
            for batch in range(int(len(dataloader.dataset)
                                   /config.BATCH_SIZE)):
                X_batch, Y_batch = self.load_unsupervised_batch(
                    dataloader, iterator)
                X_features = net(X_batch)
                X_features = X_features.cpu().detach().numpy()
                embeddings_list.append(X_features)
                label_encoder = LabelEncoder()
                labels_list.append(label_encoder.fit_transform(
                    Y_batch.cpu().detach().numpy()))
            embeddings_list = utils.convert_to_list(embeddings_list)
            labels_list = utils.convert_to_list(labels_list)
            return embeddings_list, labels_list
