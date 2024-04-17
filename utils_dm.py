import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
# import torchvision.datasets as dset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from PIL import ImageChops, Image, ImageEnhance

def convert_labels(scores_labels, normal_cls):
    # we use 1 for anomalies, and 0 for normal samples https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/datasets/mnist.py#L34
    scores_labels = [0 if y in normal_cls else 1 for y in scores_labels]
    return scores_labels

def get_equal_AD_classes(x_norm, y_norm, x_anom, y_anom):
    """
    to avoid the possible issues and bias of unbalanced datasets, an equal number of normal and anomalous samples
    will be considered in the validation and test sets (ie. a sampling over all classes is done in the majority AD class)
    """
    if x_anom.shape[0]>x_norm.shape[0]:
        nbr_samples_needed = x_norm.shape[0]
        shuffled_idx = np.arange(x_anom.shape[0])
        np.random.shuffle(shuffled_idx)
        x_anom = x_anom[shuffled_idx[:nbr_samples_needed]]
        y_anom = y_anom[shuffled_idx[:nbr_samples_needed]]
    elif x_anom.shape[0]<x_norm.shape[0]:
        nbr_samples_needed = x_anom.shape[0]
        shuffled_idx = np.arange(x_norm.shape[0])
        np.random.shuffle(shuffled_idx)
        x_norm = x_norm[shuffled_idx[:nbr_samples_needed]]
        y_norm = y_norm[shuffled_idx[:nbr_samples_needed]]
    else:
        # as many normal samples as there are anomalous samples, no modification needed
        pass
    x = np.concatenate((x_norm, x_anom))
    y = np.concatenate((y_norm, y_anom))
    shuffled_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffled_idx)
    x = x[shuffled_idx]
    y = y[shuffled_idx]
    return x, y

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def get_dataloaders_MNIST_FashionMNIST(batch_size, normal_classes, dataset_name, testvalid_ratio=0.5, seed=1):

    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST(root='./data/', train=True, download=True, transform=transform) # MNIST:60000
        test_set = MNIST(root='./data/', train=False, download=True, transform=transform) # MNIST:10000
        x_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        x_test = test_set.data.numpy() # ndarray:(10000,28,28)
        y_test = test_set.targets.numpy()
    elif dataset_name == 'fashion-mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = FashionMNIST(root='./data/', train=True, download=True, transform=transform)
        test_set = FashionMNIST(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        x_test = test_set.data.numpy()
        y_test = test_set.targets.numpy()
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = CIFAR10(root='./data/', train=True, download=True, transform=transform)
        test_set = CIFAR10(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data  # ndarray:(50000,32,32,3)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train = np.asarray(train_set.targets)
        x_test = test_set.data # ndarray:(10000,32,32,3)
        x_test = np.transpose(x_test, (0, 3, 1, 2)) # ndarray:(10000,3,32,32)
        y_test = np.asarray(test_set.targets)
    elif dataset_name == 'cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_set = CIFAR100(root='./data/', train=True, download=True, transform=transform_train)
        test_set = CIFAR100(root='./data/', train=False, download=True, transform=transform_test)
        x_train = train_set.data  # ndarray:(50000,32,32,3)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train = np.asarray(sparse2coarse(train_set.targets))
        x_test = test_set.data # ndarray:(10000,32,32,3)
        x_test = np.transpose(x_test, (0, 3, 1, 2)) # ndarray:(10000,3,32,32)
        y_test = np.asarray(sparse2coarse(test_set.targets))
    else:
        raise ValueError('Dataset not implemented in get_dataloaders_MNIST_FashionMNIST() !')

    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=testvalid_ratio, random_state=seed) # 按照testvalid_ratio(0.5)将测试集分成测试集和验证集 ndarray:(5000,28,28)

    idx_norm_train = np.zeros_like(y_train) # ndarray:(60000,)
    idx_norm_valid = np.zeros_like(y_valid) # ndarray:(5000,)
    idx_norm_test = np.zeros_like(y_test) # ndarray:(5000,)
    for normcls in normal_classes:
        idx_norm_train += np.where(y_train == normcls, 1, 0)
        idx_norm_valid += np.where(y_valid == normcls, 1, 0)
        idx_norm_test += np.where(y_test == normcls, 1, 0)

    x_train = x_train[idx_norm_train!=0]
    y_train = y_train[idx_norm_train!=0]

    # augmented_image
    offset_value = 3.5
    angle = 25
    augumented = []
    for i in range(x_train.shape[0]):
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            img = torch.from_numpy(x_train[i])  # Tensor:(28,28) (3,32,32)
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')  # Image
        elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
            img = torch.from_numpy(x_train[i]).permute((1, 2, 0))  # Tensor:(28,28) (32,32,3)
            img = Image.fromarray(img.numpy(), mode='RGB')  # Image

        augmented_image = img.copy().rotate(np.random.random() * 2 * angle - angle)
        offset1 = np.random.randint(-offset_value, offset_value)
        offset2 = np.random.randint(-offset_value, offset_value)
        augmented_image = ImageChops.offset(augmented_image, offset1, offset2)
        augmented_image = ImageEnhance.Contrast(augmented_image).enhance(1.2)
        augmented_image = torch.tensor(np.array(augmented_image)).unsqueeze(0) / 255.0  # Tensor(1,28,28)
        augumented.append(augmented_image)
    augumented = torch.cat(augumented, 0) # Tensor(2500,32,32,3)
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        augumented = augumented.permute((0,3,1,2)) # Tensor(2500,3,32,32)

    x_valid_norm = x_valid[idx_norm_valid != 0]
    y_valid_norm = y_valid[idx_norm_valid != 0]
    x_test_norm = x_test[idx_norm_test != 0]
    y_test_norm = y_test[idx_norm_test != 0]

    x_valid_anom = x_valid[idx_norm_valid == 0]
    y_valid_anom = y_valid[idx_norm_valid == 0]
    x_test_anom = x_test[idx_norm_test == 0]
    y_test_anom = y_test[idx_norm_test == 0]

    x_valid, y_valid = get_equal_AD_classes(x_valid_norm, y_valid_norm, x_valid_anom, y_valid_anom)
    x_test, y_test = get_equal_AD_classes(x_test_norm, y_test_norm, x_test_anom, y_test_anom)

    train_set = TensorDataset(torch.from_numpy(x_train/255).float(), augumented, torch.from_numpy(y_train))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = TensorDataset(torch.from_numpy(x_valid/255).float(), torch.from_numpy(y_valid)) # TensorDataset:498
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True) # DataLoader:8

    test_set = TensorDataset(torch.from_numpy(x_test/255).float(), torch.from_numpy(y_test)) # TensorDataset:502
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True) # DataLoader:8

    return train_loader, valid_loader, test_loader, x_train, y_train, x_valid, y_valid, x_test, y_test

def get_dataloaders_FMNIST(batch_size, normal_classes, dataset_name, testvalid_ratio=0.5, seed=1,ab_ratio=0):

    if  dataset_name == 'fashion-mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = FashionMNIST(root='./data/', train=True, download=True, transform=transform)
        test_set = FashionMNIST(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        x_test = test_set.data.numpy()
        y_test = test_set.targets.numpy()

    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=testvalid_ratio, random_state=seed) # 按照testvalid_ratio(0.5)将测试集分成测试集和验证集 ndarray:(5000,28,28)

    idx_norm_train = np.zeros_like(y_train) # ndarray:(60000,)
    idx_norm_valid = np.zeros_like(y_valid) # ndarray:(5000,)
    idx_norm_test = np.zeros_like(y_test) # ndarray:(5000,)
    for normcls in normal_classes:
        idx_norm_train += np.where(y_train == normcls, 1, 0)
        idx_norm_valid += np.where(y_valid == normcls, 1, 0)
        idx_norm_test += np.where(y_test == normcls, 1, 0)

    x_train_nor = x_train[idx_norm_train != 0]  # ndarray:(6000,28,28)
    y_train_nor = y_train[idx_norm_train != 0]  # ndarray:(6000,)
    x_train_ab = x_train[idx_norm_train == 0]
    x_train_ab = x_train_ab[:int(6000 * ab_ratio)]  # ndarray:(300,28,28)
    x_train_nor = np.concatenate((x_train_ab, x_train_nor))
    x_train_nor = np.random.permutation(x_train_nor)
    y_train = np.pad(y_train_nor, (0, int(6000 * ab_ratio)), 'edge')  # ndarray:(6300,)

    # augmented_image
    offset_value = 3.5
    angle = 25
    augumented = []
    for i in range(x_train_nor.shape[0]):
        if dataset_name == 'fashion-mnist':
            img = torch.from_numpy(x_train_nor[i])  # Tensor:(28,28)
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')  # Image

        augmented_image = img.copy().rotate(np.random.random() * 2 * angle - angle)
        offset1 = np.random.randint(-offset_value, offset_value)
        offset2 = np.random.randint(-offset_value, offset_value)
        augmented_image = ImageChops.offset(augmented_image, offset1, offset2)
        augmented_image = ImageEnhance.Contrast(augmented_image).enhance(1.2)
        augmented_image = torch.tensor(np.array(augmented_image)).unsqueeze(0) / 255.0  # Tensor(1,28,28)
        augumented.append(augmented_image)
    augumented = torch.cat(augumented, 0)

    x_valid_norm = x_valid[idx_norm_valid != 0]
    y_valid_norm = y_valid[idx_norm_valid != 0]
    x_test_norm = x_test[idx_norm_test != 0]
    y_test_norm = y_test[idx_norm_test != 0]

    x_valid_anom = x_valid[idx_norm_valid == 0]
    y_valid_anom = y_valid[idx_norm_valid == 0]
    x_test_anom = x_test[idx_norm_test == 0]
    y_test_anom = y_test[idx_norm_test == 0]

    x_valid, y_valid = get_equal_AD_classes(x_valid_norm, y_valid_norm, x_valid_anom, y_valid_anom)
    x_test, y_test = get_equal_AD_classes(x_test_norm, y_test_norm, x_test_anom, y_test_anom)

    train_set = TensorDataset(torch.from_numpy(x_train_nor/255).float(), augumented, torch.from_numpy(y_train))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = TensorDataset(torch.from_numpy(x_valid/255).float(), torch.from_numpy(y_valid))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    test_set = TensorDataset(torch.from_numpy(x_test/255).float(), torch.from_numpy(y_test))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, x_train_nor, y_train, x_valid, y_valid, x_test, y_test


def init_network_weights_from_pretraining(net, dataset, normal_cls, seed):
    net_dict = net.state_dict()
    ae_net_dict = torch.load('./trained_models/{}_{}_{}.pt'.format(dataset, normal_cls, seed))
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    net_dict.update(ae_net_dict)
    net.load_state_dict(net_dict)
    return net

def save_pretrained_weights(net, dataset, normal_cls, seed):
    torch.save(net.state_dict(),'./trained_models/{}_{}_{}.pt'.format(dataset, normal_cls, seed))

figure_save_path = ".\dm-fig"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)


def plot_distribution(data_loader, net, hyperspheres_center, device, status, normal_cls):
    net.eval()
    with torch.no_grad():
        for data, targets in data_loader: # 0-9
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
                complete_labels = torch.cat((complete_labels, labels), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs
                complete_labels = labels

    if len(hyperspheres_center.size())==1:
        hyperspheres_center = hyperspheres_center.unsqueeze(0)
    center_labels = torch.ones((hyperspheres_center.size()[0])).to(device)*10
    complete_outputs = torch.cat([complete_outputs, hyperspheres_center], dim=0)
    complete_labels = torch.cat([complete_labels, center_labels], dim=0)

    inputs = complete_outputs.cpu().detach().numpy()
    n_components = 2
    latent_2Ds = []
    latent_2D_names = []
    latent_2Ds.append(TSNE(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("TSNE")
    latent_2Ds.append(PCA(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("PCA")
    latent_2Ds.append(SparseRandomProjection(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("SRP")
    latent_2Ds.append(Isomap(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("ISOMAP")
    latent_2Ds.append(LocallyLinearEmbedding(n_components=n_components).fit_transform(inputs)) # latent_2Ds={list:5} 每个list：ndarray:(3166,2) 3066test+100center
    latent_2D_names.append("LLE")

    c_dict = {0: 'deeppink', 1: 'red', 2: 'orangered', 3: 'chocolate', 4: 'yellowgreen',
              5: 'chartreuse', 6: 'green', 7: 'dodgerblue', 8: 'blue', 9: 'darkviolet', 10: 'black'}

    c_dict_AD = {0: 'red', 1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'red', 6: 'red', 7: 'red', 8: 'red',
                 9: 'red', 10: 'black'}
    label_dict_AD = {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '2'}
    for normal_class in normal_cls:
        c_dict_AD[normal_class] = 'green'
        label_dict_AD[normal_class] = '0'

    fig, axs = plt.subplots(1, 5, figsize=(25, 10))
    for fig_idx in range(len(latent_2Ds)):
        x = latent_2Ds[fig_idx][:,0]
        y = latent_2Ds[fig_idx][:,1]
        for label in range(11):
            bool_array = (complete_labels == label).cpu().numpy()
            if label == 10:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict[label], label=label, alpha=1, s=100)
            else:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict[label], label=label, alpha=0.1, s=25)
        axs[fig_idx].legend()
        axs[fig_idx].set_title("{} - Normal: {} - {}".format(latent_2D_names[fig_idx], normal_cls, status))
    plt.savefig(os.path.join(figure_save_path, 'c_dict-{}.png'.format(status)))
    plt.show()

    fig, axs = plt.subplots(1, 5, figsize=(25, 10))
    for fig_idx in range(len(latent_2Ds)):
        x = latent_2Ds[fig_idx][:, 0]
        y = latent_2Ds[fig_idx][:, 1]
        for label in range(11):
            bool_array = (complete_labels == label).cpu().numpy()
            if label == 10:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict_AD[label], label=label_dict_AD[label], alpha=1, s=100)
            else:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict_AD[label], label=label_dict_AD[label], alpha=0.1, s=25)
        axs[fig_idx].legend()
        axs[fig_idx].set_title("{} - Normal: {} - {}".format(latent_2D_names[fig_idx], normal_cls, status))
    plt.savefig(os.path.join(figure_save_path, 'c_dict_AD-{}.png'.format(status)))
    plt.show()