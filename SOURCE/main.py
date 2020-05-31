# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:29:08 2020

@author: Ashima  
"""

import os
import data
import model
import config
import torch
import utils
import network
# import network_conv as network
# import network_v2 as network
# import network_conv_v1 as network

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # Initialize seed value.
    utils.set_seed_value(config.SEED)

    # Read training data. 
    train_data = data.Data()
    train_data.read(train=True)
    print("Train Data Loaded")

    if torch.cuda.is_available():
        print("Using CUDA")

    # Build model.
    generator_net = network.Generator_Network(config.BATCH_SIZE).to(
        config.device)
    generator_net.apply(utils.generator_weights_init)
    print("Generator Model Initialized")

    discriminator_net = network.Discriminator_Network().to(config.device)
    discriminator_net.apply(utils.discriminator_weights_init)
    print("Discriminator Model Initialized")
    
    print("Discriminator model...")
    print(discriminator_net)
    
    modeloperator = model.Operators(generator_net, discriminator_net)
    print("Model Built")
    
    modeloperator.pretrain(train_data)
    print("Model Pretrained")

    modeloperator.train(train_data)
    print("Model Trained")

    # Read test data.
    test_data = data.Data()
    test_data.read(train=False)
    print("Test data loaded")

    # Evaluate model on KNN Classifier.
    modeloperator.test(train_data, test_data, discriminator_net)
    print("Calculated accuracy on test data")

    # Plot t-SNE to visualize the latent space.
    utils.plot_latent_tsne(train_data, discriminator_net, plot_type="TRAIN")
    print("Plotted train dataset tSNE of Latent Space")

    utils.plot_latent_tsne(test_data, discriminator_net, plot_type="TEST")
    print("Plotting test dataset tSNE of Latent Space")
