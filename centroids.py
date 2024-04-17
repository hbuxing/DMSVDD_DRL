import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np

def init_center_c_mode_mean(device, train_loader, net, eps=0.1, regularize=True):
    """
    Initialize hypersphere center c as the regularized means from an initial forward pass on the data. One centroid
    is shared by all normal data & classes. This is the deep SVDD baseline.
    """

    net.eval()
    with torch.no_grad():
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=device)
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
        c /= n_samples

        if regularize:
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

    return c

def init_centers_c_kmeans_MSVDD(device, train_loader, net, nbr_centroids=3, eps=0.1, batch_size=128, seed=1): # x_train：18623 nbr_centroids=100 batch_size=512 seed=0
    """
    Different function for MSVDD paper implementation.
    """

    # train_set = TensorDataset(torch.from_numpy(x_train/255).float(), torch.from_numpy(y_train))
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    net.eval()
    with torch.no_grad():
        for index, (x, _, y) in enumerate(train_loader):
            x = x.unsqueeze(dim=1)
            x, y = x.to(device), y.to(device)
            output, _ = net(x)
            try:
                complete_outputs = torch.cat((complete_outputs, output[0]), dim=0)
            except UnboundLocalError:
                complete_outputs = output[0]

    complete_outputs = complete_outputs.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=nbr_centroids, random_state=seed).fit(complete_outputs)

    return torch.from_numpy(kmeans.cluster_centers_).to(device)

def init_centers_c_kmeans_MSVDD_cifar(device, train_loader, net, nbr_centroids=3, eps=0.1, batch_size=128, seed=1): # x_train：18623 nbr_centroids=100 batch_size=512 seed=0
    """
    Different function for MSVDD paper implementation.
    """

    # train_set = TensorDataset(torch.from_numpy(x_train/255).float(), torch.from_numpy(y_train))
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    net.eval()
    with torch.no_grad():
        for index, (x, _, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            output, _= net(x)
            try:
                complete_outputs = torch.cat((complete_outputs, output[0]), dim=0)
            except UnboundLocalError:
                complete_outputs = output[0]

    complete_outputs = complete_outputs.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=nbr_centroids, random_state=seed).fit(complete_outputs)

    return torch.from_numpy(kmeans.cluster_centers_).to(device)

def filter_centers_DMSVDD(hyperspheres_center, radius):
    return hyperspheres_center[radius > 0.0]

def update_radius_DMSVDD(hyperspheres_center, nu, train_loader, net, device): # nu=0.1
    net.eval()
    with torch.no_grad():
        for index, (x, _, y) in enumerate(train_loader):
            x = x.unsqueeze(dim=1)
            x, y = x.to(device), y.to(device)
            outputs, _= net(x)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs[0]), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs[0]

    # now that populated-enough centers are updated, update the associated radius
    dist_to_centers = torch.cdist(complete_outputs, hyperspheres_center)
    dist_to_best_center, best_center_idx = torch.min(dist_to_centers, dim=1)
    centers, centers_occurrence = torch.unique(best_center_idx, return_counts=True)

    # handle centers with zero population that disappeared due to the torch.unique() 处理出现次数为0的中心
    new_centers_occurrence = torch.zeros((hyperspheres_center.size()[0],)).long().to(device)
    new_centers_occurrence[centers] = centers_occurrence
    centers_occurrence = new_centers_occurrence

    good_centers = centers_occurrence > nu * torch.max(centers_occurrence)

    radius = torch.zeros((hyperspheres_center.size()[0],)).to(device)
    for center_idx in range(hyperspheres_center.size()[0]):
        try:
            radius[center_idx] = torch.quantile(dist_to_centers[best_center_idx == center_idx, center_idx],q=1-nu) # 1-nu分位数
        except RuntimeError: # handle centroids without samples, which can't yield any quantile
            radius[center_idx] = 0.0

    radius[~good_centers] = 0.0 # centroids with samples but not enough

    net.train()
    return radius

def update_radius_DMSVDD_cifar(hyperspheres_center, nu, train_loader, net, device): # nu=0.1
    net.eval()
    with torch.no_grad():
        for index, (x, _, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            outputs, _= net(x)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs[0]), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs[0]

    # now that populated-enough centers are updated, update the associated radius
    dist_to_centers = torch.cdist(complete_outputs, hyperspheres_center)
    dist_to_best_center, best_center_idx = torch.min(dist_to_centers, dim=1)
    centers, centers_occurrence = torch.unique(best_center_idx, return_counts=True)

    # handle centers with zero population that disappeared due to the torch.unique()
    new_centers_occurrence = torch.zeros((hyperspheres_center.size()[0],)).long().to(device)
    new_centers_occurrence[centers] = centers_occurrence
    centers_occurrence = new_centers_occurrence

    good_centers = centers_occurrence > nu * torch.max(centers_occurrence)

    radius = torch.zeros((hyperspheres_center.size()[0],)).to(device)
    for center_idx in range(hyperspheres_center.size()[0]):
        try:
            radius[center_idx] = torch.quantile(dist_to_centers[best_center_idx == center_idx, center_idx],q=1-nu)
        except RuntimeError: # handle centroids without samples, which can't yield any quantile
            radius[center_idx] = 0.0

    radius[~good_centers] = 0.0 # centroids with samples but not enough

    net.train()
    return radius