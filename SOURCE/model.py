# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:29:08 2020

@author: Ashima
"""

import time
import torch
import utils
import config
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score


class Operators():
    """Train generator and discriminator in adversarial fashion
    and test the trained discriminator using K-NN search."""

    def __init__(self, gen_net, disc_net):
        self.gen_net = gen_net
        self.disc_net = disc_net
        self.loss = nn.BCELoss()
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(),
                                              lr=config.PRETRAIN_LEARNING_RATE,
                                              betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.disc_net.parameters(),
                                              lr=config.PRETRAIN_LEARNING_RATE,
                                              betas=(0.5, 0.999))

    def compute_labeled_loss(self, a, b, c):
        """Calculate labeled loss."""
        n_positive = torch.sqrt(torch.sum((a - b)**2, axis=1))
        n_negative = torch.sqrt(torch.sum((a - c)**2, axis=1))
        z = torch.cat([n_negative.view(n_negative.size()[0], 1),
                       n_positive.view(n_negative.size()[0], 1)], axis=1)
        z = utils.log_sum_exp(z, axis=1)
        return n_positive, n_negative, z

    def pretrain(self, data):
        """Pretrain GAN network."""
        print("Start Pretraining...")
        start_time = time.time()
        gen_losses = []
        dis_losses = []
        self.gen_net.train()
        self.disc_net.train()
        num_batches = int(len(data.unlabeled_dataloader_disc.dataset)
                          /config.BATCH_SIZE)
        print("Num batches: ", num_batches)
        for epoch in range(config.NUM_PRETRAIN_EPOCHS):
            G_losses = []
            D_losses = []
            for batch in range(num_batches):
                X_batch, _ = data.load_unsupervised_batch(
                data.unlabeled_dataloader_disc, data.unlabeled_iterator_disc)

                # Update discriminator.
                dis_loss_ = self.pretrain_discriminator(X_batch)
                D_losses.append(dis_loss_)

                # Update generator.
                gen_loss_ = self.train_generator(X_batch)
                G_losses.append(gen_loss_)

            if epoch % 1 == 0:
                print('Epoch: %d, Generator Loss: %.3f, ' \
                      'Discriminator Loss: %.3f' % (epoch, np.mean(G_losses),
                      np.mean(D_losses)))

            if epoch % config.PRINT_FREQUENCY == 0:
                self.visualize(epoch)

            gen_losses.append(np.mean(G_losses))
            dis_losses.append(np.mean(D_losses))

        print("Total pretraining time: ", time.time() - start_time)
        torch.save(self.gen_net.state_dict(),
                   utils.generator_pretrained_path())
        torch.save(self.disc_net.state_dict(),
                   utils.discriminator_pretrained_path())
        utils.plot_pretrain(gen_losses, dis_losses, config.NUM_PRETRAIN_EPOCHS)

    def train(self, data):
        """Train GAN network."""
        start_time = time.time()

        if config.USE_PRETRAINED:
            self.gen_net.load_state_dict(torch.load(
                utils.generator_pretrained_path()))
            self.disc_net.load_state_dict(torch.load(
                utils.discriminator_pretrained_path()))
            
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(),
                                              lr=config.LEARNING_RATE,
                                              betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.disc_net.parameters(),
                                              lr=config.LEARNING_RATE,
                                              betas=(0.5, 0.999))

        gen_losses = []
        labeled_losses_list = []
        unlabeled_losses_list = []
        self.gen_net.train()
        self.disc_net.train()
        print("Num batches: ", int(len(
            data.unlabeled_dataloader_disc.dataset)/config.BATCH_SIZE))
        for epoch in range(config.NUM_EPOCHS):
            G_losses = []
            labeled_losses = []
            unlabeled_losses = []
            for batch in range(int(len(
                    data.unlabeled_dataloader_disc.dataset)/config.BATCH_SIZE)):
                labeled_X = data.load_triplet_batch(
                    data.labeled_dataloader, data.labeled_iterator)
                unlabeled_X_disc, _ = data.load_unsupervised_batch(
                data.unlabeled_dataloader_disc, data.unlabeled_iterator_disc)
                unlabeled_X_gen, _ = data.load_unsupervised_batch(
                    data.unlabeled_dataloader_gen, data.unlabeled_iterator_gen)

                # Update discriminator.
                labeled_loss, unlabeled_loss = self.train_discriminator(
                    labeled_X, unlabeled_X_disc)
                labeled_losses.append(labeled_loss)
                unlabeled_losses.append(unlabeled_loss)

                # Update generator.
                gen_loss = self.train_generator(unlabeled_X_gen)
                G_losses.append(gen_loss)

            if epoch % 1 == 0:
                print('Epoch: %d, Generator Loss: %.3f, ' \
                      'Labeled Loss: %.3f, Unlabeled Loss : %.3f' % (
                    epoch, np.mean(G_losses), np.mean(labeled_losses),
                    np.mean(unlabeled_losses)))

            if epoch % config.PRINT_FREQUENCY == 0:
                self.visualize(epoch)

            gen_losses.append(np.mean(G_losses))
            labeled_losses_list.append(np.mean(labeled_losses))
            unlabeled_losses_list.append(np.mean(unlabeled_losses))

        print("Total training time: ", time.time() - start_time)
        torch.save(self.gen_net.state_dict(), utils.generator_path())
        torch.save(self.disc_net.state_dict(), utils.discriminator_path())
        utils.plot_train(gen_losses, labeled_losses_list,
                         unlabeled_losses_list, config.NUM_EPOCHS)

    def pretrain_discriminator(self, unlabeled_X):
        """Pretrain discriminator in unsupervised fashion and
        calculate discriminator based loss.
        Return only unsupervised loss.
        """
        self.disc_optimizer.zero_grad()        
        [out_fake, out_unlabeled] = [self.disc_net(self.gen_net(
            unlabeled_X.size()[0]).view(unlabeled_X.size())),
            self.disc_net(unlabeled_X)]                       # D(G(z)) , D(x)
        logz_unlabel, logz_fake = [utils.log_sum_exp(out_unlabeled),
                                  utils.log_sum_exp(out_fake)] # log ∑e^x_i

        unlabeled_loss = 0.5 * (-torch.mean(logz_unlabel) +
                                torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        disc_loss = unlabeled_loss
        disc_loss.backward()
        self.disc_optimizer.step()
        return unlabeled_loss.item()

    def train_discriminator(self, labeled_X, unlabeled_X):
        """Train discriminator in semi-supervised fashion and
        compute both labeled and unlabeled losses.
        Return both supervised and unsupervised loss.
        """
        self.disc_optimizer.zero_grad()
        [out_anchor, out_pos, out_neg] = [self.disc_net(labeled_X[0]),
                                          self.disc_net(labeled_X[1]),
                                          self.disc_net(labeled_X[2])]
        [out_fake, out_unlabeled] = [self.disc_net(self.gen_net(
            unlabeled_X.size()[0]).view(unlabeled_X.size())),
            self.disc_net(unlabeled_X)]

        n_positive, n_negative, z = self.compute_labeled_loss(out_anchor,
                                                       out_pos, out_neg)
        labeled_loss = - torch.mean(n_negative) + torch.mean(z)

        logz_unlabel, logz_fake = [utils.log_sum_exp(out_unlabeled),
                                  utils.log_sum_exp(out_fake)] # log ∑e^x_i

        unlabeled_loss = 0.5 * (-torch.mean(logz_unlabel) +
                                torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        disc_loss = labeled_loss + unlabeled_loss
        disc_loss.backward()
        self.disc_optimizer.step()
        return labeled_loss.item(), unlabeled_loss.item()

    def train_generator(self, unlabeled_X):
        """Train generator and compute its losses."""
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        out_unlabeled = self.gen_net(unlabeled_X.size()[0]).view(
            unlabeled_X.size())                                          # G(z)     
        mom_gen, output_fake = self.disc_net(out_unlabeled, feature=True)# f(G(z))
        mom_unlabel, _ = self.disc_net(unlabeled_X, feature=True)        # f(x)
        mom_gen = torch.mean(mom_gen, dim=0)
        mom_unlabel = torch.mean(mom_unlabel, dim=0)
        gen_loss = torch.mean((mom_gen - mom_unlabel) ** 2)
        # gen_loss = loss_fm
        gen_loss.backward()
        self.gen_optimizer.step()
        return gen_loss.item()

    def test(self, train_data, test_data, net):
        """Test trained discriminator using K-Nearest Neighbor Search."""
        X_train, Y_train = train_data.load_features(net, string="train")
        print("No of examples to train: ", X_train.shape[0])
        start_time = time.time()
        self.classifier = KNeighborsClassifier(n_neighbors=config.NEIGHBOURS)
        self.classifier.fit(X_train, Y_train)
        print("Training time: ", time.time() - start_time)
        print("Train Accuracy score: ", self.classifier.score(X_train, Y_train))
        Y_train = label_binarize(Y_train, classes=config.CLASS_LABELS)
        Y_train_pred = label_binarize(self.classifier.predict(X_train),
                                      classes=config.CLASS_LABELS)
        print("Train average precision (mAP): ", average_precision_score(
            Y_train, Y_train_pred))

        X_test, Y_test = test_data.load_features(net, string="test")
        print("Test Accuracy score: ", self.classifier.score(X_test, Y_test))
        Y_test = label_binarize(Y_test, classes=config.CLASS_LABELS)
        Y_test_pred = label_binarize(self.classifier.predict(X_test),
                                     classes=config.CLASS_LABELS)
        print("Test average precision (mAP): ", average_precision_score(
            Y_test, Y_test_pred))
        print("Total time taken for evaluation on {}-nn is {}: ".format(
            config.NEIGHBOURS, (time.time() - start_time)))

    def visualize(self, epoch):
        self.gen_net.eval()
        with torch.no_grad():
            test_images = self.gen_net(config.BATCH_SIZE).cpu().detach().numpy()
        self.gen_net.train()
        print("test images shape: ", len(test_images))
        grid_size = 5
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in itertools.product(range(grid_size), range(grid_size)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(config.OUT_DIR + 'Generated_Images_GANS_'+
                    str(epoch)+'.jpg')
        plt.show()
