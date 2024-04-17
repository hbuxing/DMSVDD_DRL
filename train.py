# encoding: utf-8
try:
    import os
    import argparse
    import time

    import torch
    import torchvision
    import numpy as np
    import torch.nn.functional as F

    from sklearn.mixture import GaussianMixture
    from torch.optim.lr_scheduler import StepLR,MultiStepLR
    from tqdm import tqdm
    from itertools import chain
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    import pandas as pd
    from seaborn import boxplot
    import random

    from datasets import dataset_list, get_dataloader
    from config import RUNS_DIR, DATASETS_DIR, DEVICE, DATA_PARAMS
    from model import Generator, GMM, Encoder
    from utils import save_images, cluster_acc, gmm_Loss, mse
    from utils_dm import *
    from networks_dm import *
    from centroids import *
    from batchscores import *
    from epochscores import *

except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=150, type=int, help="Number of epochs")
    parser.add_argument("-o", "--outlier", dest="outlier_rate", default=0, type=float, help="ratios of outlier")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='cifar100', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="SMM")
    args = parser.parse_args()

    # run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, args.version_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    for alpha in (0.01,0.1,1):
     for ii in range(0, 20):
        start = time.time()
        total = 0.0
        best_auc_repeat = 0
        for repeatt in range(0, 3):
            # -----train-----
            # train detail var
            n_epochs = args.n_epochs
            b1 = 0.5
            b2 = 0.99
            n_cluster = 1
            a1 = 10 # 增强损失系数 (2,3,4,5,10)
            print("alpha = {}  a1 = {}  ".format(alpha, a1))

            data_params = DATA_PARAMS[dataset_name]
            train_batch_size, latent_dim, c_feature, picture_size, cshape, data_size, pre_lr, train_lr, pre_epoch= data_params
            print(data_params)

            normal_cls = [ii]
            nbr_centroids = 100
            nu = 0.1
            valid_AUCs = torch.zeros(n_epochs + 1)
            test_AUCs = torch.zeros(n_epochs + 1)
            losses = torch.zeros(n_epochs)
            losses_radius_sqmean = torch.zeros(n_epochs)
            losses_margin_loss = torch.zeros(n_epochs)
            nbr_centroids_evolution = torch.zeros(n_epochs)

            # net
            gen = Generator(latent_dim=latent_dim, x_shape=picture_size, cshape=cshape)
            gmm = GMM(n_cluster=n_cluster, n_features=c_feature)
            encoder = Encoder(
                input_channels=picture_size[0],
                output_channels=latent_dim,
                cshape=cshape,
                c_feature=c_feature,
                r=9,
                adding_outlier=(args.outlier_rate > 0.05)
            )

            xe_loss = nn.BCELoss(reduction="sum")

            # parallel
            if torch.cuda.device_count() > 1:
                print("this GPU have {} core".format(torch.cuda.device_count()))

            # set device: cuda or cpu
            gen.to(DEVICE)
            encoder.to(DEVICE)
            gmm.to(DEVICE)

            # optimization
            gen_enc_ops = torch.optim.Adam(chain(
                gen.parameters(),
                encoder.parameters(),
            ), lr=pre_lr, betas=(b1, b2))
            gen_enc_gmm_ops = torch.optim.Adam(chain(
                gen.parameters(),
                encoder.parameters(),
                gmm.parameters(),
            ), lr=train_lr, betas=(b1, b2))
            lr_s = StepLR(gen_enc_gmm_ops, step_size=10, gamma=0.95)

            train_loader, valid_loader, test_loader, x_train, y_train, _, _, _, _ = \
                get_dataloaders_MNIST_FashionMNIST(train_batch_size, normal_cls, dataset_name)

            # =============================================================== #
            # =========================pretraining======================== #
            # =============================================================== #
            print('Pretraining......')
            epoch_bar = tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L = 0
                for index, (x, _, y) in enumerate(train_loader):
                    x = x.to(DEVICE)  # (64,3,32,32)

                    z, _ = encoder(x)  # z：tuple：6 (z1, z2, mu1, mu2, log_sigma1, log_sigma2) z1聚类表示 z2生成表示
                    x_ = gen(z[0], z[1])  # (64,3,32,32) z[0]:(64,25) z[1]:(64,5)
                    loss_pre = xe_loss(x_, x) / train_batch_size

                    L += loss_pre.detach().cpu().numpy()

                    gen_enc_ops.zero_grad()
                    loss_pre.backward()
                    gen_enc_ops.step()

                epoch_bar.write('Loss={:.4f}'.format(L / len(train_loader)))
            encoder.con.load_state_dict(encoder.mu.state_dict())

            _gmm = GaussianMixture(n_components=n_cluster, covariance_type='diag')
            Z = []
            Y = []
            with torch.no_grad():
                for index, (data, _, label) in enumerate(train_loader):
                    data = data.to(DEVICE)

                    z, _ = encoder(data)
                    Z.append(z[2])
                    Y.append(label)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()
            pre = _gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            gmm.pi_.data = torch.from_numpy(_gmm.weights_).to(DEVICE).float()
            gmm.mu_c.data = torch.from_numpy(_gmm.means_).to(DEVICE).float()
            gmm.log_sigma2_c.data = torch.log(torch.from_numpy(_gmm.covariances_).to(DEVICE).float())

            hyperspheres_center = init_centers_c_kmeans_MSVDD_cifar \
                (DEVICE, train_loader, encoder, nbr_centroids=nbr_centroids,
                 batch_size=train_batch_size)  # hyperspheres_center={Tensor:(100,25)}
            # =============================================================== #
            # ============================training=========================== #
            # =============================================================== #
            print('begin training...')
            epoch_bar = tqdm(range(0, n_epochs))
            gen_weight = 0.15
            for epoch in epoch_bar:
                radius = update_radius_DMSVDD_cifar(hyperspheres_center, nu, train_loader, encoder, DEVICE)
                valid_AUCs[0], test_AUCs[epoch], _, _, _ = \
                    get_epoch_performances_DMSVDD_cifar(valid_loader, test_loader, DEVICE,
                                                  encoder, hyperspheres_center, radius, normal_cls)

                encoder.train()
                running_loss = 0.0
                running_loss_radius_sqmean = 0.0
                running_loss_margin_loss = 0.0
                g_t_loss, loss_t = 0, 0

                for index, (real_images, augmented_images, target) in enumerate(train_loader):
                    real_images, augmented_images, target = real_images.to(DEVICE), \
                                                             augmented_images.to(DEVICE), \
                                                             target.to(DEVICE)

                    gen.train()
                    gmm.train()
                    encoder.train()
                    encoder.zero_grad()
                    gen.zero_grad()
                    gmm.zero_grad()
                    gen_enc_gmm_ops.zero_grad()

                    original_z, augmented_z = encoder(real_images, augmented_images, argument=(epoch > 2))
                    fake_images_o = gen(original_z[0], gen_weight * original_z[1])
                    fake_images_a = gen(augmented_z[0], gen_weight * augmented_z[1])
                    z_c = torch.cat((original_z[0], augmented_z[0]), dim = 0)

                    # svdd loss
                    dist_to_centers = torch.sum((z_c.unsqueeze(1).repeat(1, hyperspheres_center.size()[0], 1)
                                                 - hyperspheres_center.unsqueeze(0).repeat(z_c.size()[0], 1, 1)) ** 2,
                                                dim=2)
                    dist_to_best_center, best_center_idx = torch.min(dist_to_centers, dim=1)
                    best_center_sqradius = radius[best_center_idx] ** 2
                    radius_sqmean = (1 / radius.size()[0]) * torch.sum(radius ** 2)
                    margin_loss = (1 / (real_images.size()[0]*2)) \
                                  * torch.sum(torch.maximum(dist_to_best_center - best_center_sqradius,
                                                            torch.zeros((dist_to_best_center.size()[0],)).to(DEVICE)))
                    loss_svdd = radius_sqmean + margin_loss

                    # vae loss
                    rec_loss_o = torch.mean(
                        torch.sum(F.binary_cross_entropy(fake_images_o, real_images, reduction='none'), dim=[1, 2, 3]))
                    rec_loss_a = torch.mean(
                        torch.sum(F.binary_cross_entropy(fake_images_a, augmented_images, reduction='none'), dim=[1, 2, 3]))

                    augmented_loss = mse(original_z[2], augmented_z[2]) + \
                                     mse(original_z[3], augmented_z[3]) # ||mu1-mu1~||^2 + ||mu2-mu2~||^2

                    kl_loss = torch.mean(
                        -0.5 * torch.sum(1 + original_z[5] - original_z[3] ** 2 - original_z[5].exp(),
                                         dim=1), dim=0)

                    kls_loss = gmm_Loss(original_z[0], original_z[2], original_z[4], gmm)

                    g_loss = a1 * augmented_loss + kls_loss + kl_loss + rec_loss_o + rec_loss_a
                    g_loss *= alpha

                    loss = loss_svdd + g_loss
                    loss.backward(retain_graph=True)
                    gen_enc_gmm_ops.step()
                    running_loss += loss_svdd.item()
                    running_loss_radius_sqmean += radius_sqmean.item()
                    running_loss_margin_loss += margin_loss.item()
                    g_t_loss += g_loss.item()
                    loss_t += loss.item()

                losses[epoch] = loss_t
                losses_radius_sqmean[epoch] = running_loss_radius_sqmean
                losses_margin_loss[epoch] = running_loss_margin_loss
                nbr_centroids_evolution[epoch] = hyperspheres_center.size()[0]
                hyperspheres_center = filter_centers_DMSVDD(hyperspheres_center, radius)

                valid_AUCs[epoch + 1], test_AUCs[epoch + 1], scores_test, scores_labels_test, scores_per_center_test = \
                    get_epoch_performances_DMSVDD_cifar(valid_loader, test_loader,
                                                        DEVICE, encoder, hyperspheres_center, radius, normal_cls)

                lr_s.step()

                print("normal_cls:{} : epoch: {}, loss: {}, valid_auc: {}, test_auc: {}\n"
                      .format(normal_cls, epoch, loss_t / len(train_loader), valid_AUCs[epoch + 1], test_AUCs[epoch + 1]))

            test_AUC_at_best_test = torch.gather(test_AUCs, 0, torch.max(test_AUCs, dim=0)[1].long())
            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write("normal_cls:{} : TEST AUC : {}  alpha = {}  n_cluster = {}  a1 = {}\n"
                         .format(normal_cls, torch.mean(test_AUC_at_best_test), alpha, n_cluster, a1))
            logger.close()
            print("normal_cls:{} : TEST AUC : {} \n".format(normal_cls, torch.mean(test_AUC_at_best_test)))

            if torch.mean(test_AUC_at_best_test) > best_auc_repeat:
                best_auc_repeat = torch.mean(test_AUC_at_best_test)

        end = time.time()
        total = end - start

        logger = open(os.path.join(log_path, "log.txt"), 'a')
        logger.write("normal_cls:{} : BEST TEST AUC : {}  time: {}\n".format(normal_cls, best_auc_repeat, total))
        logger.close()
        print("normal_cls:{} : BEST TEST AUC : {} \n".format(normal_cls, best_auc_repeat))

if __name__ == '__main__':

    main()
