# import numpy as np
import torch
from torch import Tensor
import math
# from .kernel_dim_red_trace import kernel_dim_red_trace
from .kernel_dim_by_trace import kernel_dim_by_trace


MAX_LOOP = 50  # number of iterations in KDR method
EPS = 1e-4  # regularization parameter for matrix inversion
ETA = 10.  # range of golden ratio search
ANL = 4  # maximum value for anealing


# def median_dist(x):
#     n = x.shape[0]  # number of data
#     ab = x @ x.T  # (xi.T @ xj)_ij
#     aa = np.diag(ab)
#     dx = np.repeat(aa[:, np.newaxis], n, axis=1) + np.repeat(aa[np.newaxis, :], n, axis=0) - 2*ab # shape(n,n)
#     dx -= np.diag(np.diag(dx))  # dx = xi.T @ xi + xj.T @ xj - 2*xi.T @ xj
#     dx = dx[np.nonzero(dx)]
#     return np.sqrt(np.median(dx))


def median_dist(samples):
    num = samples.shape[0]  # number of data
    ab = samples @ samples.T  # (xi.T @ xj)_ij
    aa = torch.diag(ab)
    d = torch.repeat_interleave(aa.unsqueeze(-1), num, dim=1)  # shape(n,n)
    dx = d + d.T - 2 * ab  # dx = xi.T @ xi + xj.T @ xj - 2*xi.T @ xj
    dx -= torch.diag(torch.diag(dx))
    dx = dx.view(-1)
    dx = dx[torch.nonzero(dx).squeeze()]
    return torch.sqrt(torch.median(dx))


# def kdr_sample(x, y, K):
#     """
#     Args:
#         K: dimension of effective direction
#     """
#     num, ndims = x.shape  # number of data, dim of x
#     x = (x - np.mean(x, axis=0, keepdims=True))/np.std(x, axis=0, keepdims=True, ddof=1)  # standardization of x

#     # Gaussian kernels are used. Deviation parameter are set by the median of mutual distance.
#     # In the aneaning, sigma changes to 2*median to 0.5*median
#     sigma_x = 0.5 * median_dist(x)
#     sigma_y = median_dist(y)  # As y is discrete, tuning is not necessary
#     # kdr optimization. Steepest descent with line search
#     b = kernel_dim_red_trace(x, y, K, MAX_LOOP, sigma_x*np.sqrt(K/ndims), sigma_y, EPS, ETA, ANL)
#     return b


def kdr_sample(X: Tensor, Y: Tensor, d):
    """
    Args:
        d: dimension of effective direction
    """
    num, dim = X.shape  # number of data, dim of x
    X = (X - torch.mean(X, dim=0, keepdim=True)) / torch.std(X, dim=0, keepdim=True)  # standardization of x

    # Gaussian kernels are used. Deviation parameter are set by the median of mutual distance.
    # In the aneaning, sigma changes to 2*median to 0.5*median
    sigma_x = 0.5 * median_dist(X)
    sigma_y = median_dist(Y)  # As y is discrete, tuning is not necessary
    # KDR optimization. Steepest descent with line search
    B = kernel_dim_by_trace(X, Y, d, MAX_LOOP, sigma_x * math.sqrt(d/dim), sigma_y, EPS, ETA, ANL)
    return B

