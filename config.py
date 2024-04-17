# encoding: utf-8
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = ROOT_DIR

DATASETS_DIR = os.path.join(REPO_DIR, 'data')

RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# train_batch_size, latent_dim, c_feature, picture_size, cshape, all_data_size, pre_lr, train_lr, pre_epoch
DATA_PARAMS = {
    'mnist': (64, 10, 7, (1, 28, 28), (128, 7, 7), 60000, 1e-4, 2e-3, 50),
    'fashion-mnist': (64, 15, 10, (1, 28, 28), (128, 7, 7), 60000, 1e-4, 1e-4, 50),
    'cifar10': (64, 30, 25, (3, 32, 32), (128, 8, 8), 50000, 1e-4, 1e-4, 50),
    'cifar100': (64, 30, 25, (3, 32, 32), (128, 8, 8), 50000, 1e-4, 1e-4, 50),
}

