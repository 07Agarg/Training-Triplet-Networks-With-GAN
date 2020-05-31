# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:29:08 2020

@author: Ashima
"""

# References: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

import os
import time
import torch
import config
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def set_seed_value(seed):
    """Set various seed values."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def log_sum_exp(x, axis=1):
    """Compute LogSumExp as mentioned in
    https://en.wikipedia.org/wiki/LogSumExp."""
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def generator_weights_init(m):
    """Initialize weights of generator and discriminator network."""
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.05)
        # m.bias.data.fill_(0.)


def discriminator_weights_init(m):
    """Initialize weights of generator and discriminator network."""
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0.)


def plot_pretrain(gen_loss_list, dis_loss_list, epochs):
    """Plot generator and discriminator pretraining loss curves."""
    markers = ['.', 'o']
    colors = ['r', 'b']
    x = np.arange(epochs)
    plt.plot(np.asarray(x), np.asarray(gen_loss_list),
             label="Gen_loss", color=colors[0],
             marker=markers[0])
    plt.plot(np.asarray(x), np.asarray(dis_loss_list),
             label="Dis_loss", color=colors[1],
             marker=markers[1])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Pretraining Loss Curve")
    plt.legend()
    plt.savefig(config.OUT_DIR + 'PretrainLossCurve.jpg')
    # plt.show()


def plot_train(gen_loss_list, labeled_loss_list, unlabeled_loss_list, epochs):
    """Plot generator, labeled loss and unlabeled training curves."""
    markers = ['.', 'o', '*']
    colors = ['r', 'b', 'g']
    x = np.arange(epochs)
    plt.plot(np.asarray(x), np.asarray(gen_loss_list),
             label="Gen_loss", color=colors[0],
             marker=markers[0])
    plt.plot(np.asarray(x), np.asarray(labeled_loss_list),
             label="Labeled_loss", color=colors[1],
             marker=markers[1])
    plt.plot(np.asarray(x), np.asarray(unlabeled_loss_list),
             label="Unlabeled_loss", color=colors[2],
             marker=markers[2])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(config.OUT_DIR + 'TrainLossCurve.jpg')
    # plt.show()
    

def plot(loss_list, epochs, plot_label):
    """Plot individual loss curves."""
    markers = ['.']
    colors = ['r']
    x = np.arange(epochs)
    plt.plot(np.asarray(x), np.asarray(loss_list),
             label=plot_label, color=colors[0],
             marker=markers[0])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Training Loss Curve "+plot_label)
    plt.legend()
    plt.savefig(config.OUT_DIR + 'TrainLossCurve_'+plot_label+'.jpg')
    # plt.show()


def plot_tsne(data_subset, df, string):
    """Plot t-SNE."""
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hsv", config.NUM_CLASSES),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig(config.OUT_DIR + string + '.jpg')


def plot_latent_tsne(data, net, plot_type="TRAIN"):
    """Helper function to plot tsne of latent space."""
    net.load_state_dict(torch.load(discriminator_path()))
    net.eval()
    with torch.no_grad():
        x = data.dataset.data.type(torch.FloatTensor)[:8000]
        y = data.dataset.targets[:8000]
    encoded = net(x.to(config.device))
    X = encoded.view(encoded.size()[0], -1).cpu().detach().numpy()
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))
    data_subset = df[feat_cols].values
    plot_tsne(data_subset, df, "LatentTSNE_"+plot_type)


def convert_to_list(data):
    """Convert list of lists to list."""
    final_list = []
    for i in data:
        try:
            for j in i:
                final_list.append(j)
        except:
            print("ERROR")
    final_list = np.array(final_list)
    return final_list


def generator_pretrained_path():
    """Return generator's pretrained path."""
    path = os.path.join(config.MODEL_DIR,
                        "model_gen" +
                        str(config.BATCH_SIZE) +
                        "_" +
                        str(config.NUM_PRETRAIN_EPOCHS) +
                        ".pt")
    print("Generator pretrained path: ", path)
    return path


def discriminator_pretrained_path():
    """Return discriminator's pretrained path."""
    path = os.path.join(config.MODEL_DIR,
                        "model_disc" +
                        str(config.BATCH_SIZE) +
                        "_" +
                        str(config.NUM_PRETRAIN_EPOCHS) +
                        ".pt")
    print("Discriminator pretrained path: ", path)
    return path


def generator_path():
    """Return generator path."""
    path = os.path.join(config.MODEL_DIR,
                        "model_gen" +
                        str(config.BATCH_SIZE) +
                        "_" +
                        str(config.NUM_EPOCHS) +
                        ".pt")
    print("Generator path: ", path)
    return path


def discriminator_path():
    """Return discriminator path."""
    path = os.path.join(config.MODEL_DIR,
                        "model_disc" +
                        str(config.BATCH_SIZE) +
                        "_" +
                        str(config.NUM_EPOCHS) +
                        ".pt")
    print("Discriminator path: ", path)
    return path
