# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:29:08 2020

@author: Ashima
"""
import os
import torch

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 28*28
BATCH_SIZE = 100
NUM_FEATURES = 16
NUM_CLASSES = 10
NUM_LABELED_SAMPLES_PER_CLASS = 10
NUM_TRIPLETS = 90000
NEIGHBOURS = 9
CLASS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = True
NUM_PRETRAIN_EPOCHS = 100
NUM_EPOCHS = 30
PRETRAIN_LEARNING_RATE = 0.0003
LEARNING_RATE = 0.0003
TEST_SAMPLES = 25

PRINT_FREQUENCY = 10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
                                                                        'cpu')
