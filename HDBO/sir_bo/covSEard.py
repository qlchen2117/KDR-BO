import numpy as np
from .sq_dist import sq_dist


def cov_se_ard(hyp: np.ndarray, xx: np.ndarray, zz: np.ndarray=None):
    """
    Squared Exponential covariance function with Automatic Relevance Determination (ARD)
    distance measure. The covariance function is parameterized as:
        k(x^p,x^q) = sf2 * exp(-(x^p - x^q).T @ inv(P) @ (x^p - x^q) / 2)
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space and sf2 is the signal variance.
    The hyper-parameters are:
        hyp = [ log(ell_1), log(ell_2), ..., log(ell_D), log(sqrt(sf2)) ]
    """

    _, dim = xx.shape
    ell = np.exp(hyp[:dim])  # characteristic length scale
    sf2 = np.exp(2 * hyp[dim])  # signal variance

    # precompute squared distances
    if zz is None:
        gram_xz = sq_dist((xx * (1./ell)).T)  # symmetric matrix Kxx
    else:
        gram_xz = sq_dist( (xx * (1./ell)).T, (zz * (1./ell)).T)  # cross covariance Kxz

    kernel_xz = sf2 * np.exp(-gram_xz/2)  # covariance
    # grad = (np.diag(1./ell**2) @ (xx.T - zz.T)) * kernel_xz.T
    return kernel_xz
    # # derivatives
    # if i < d:  # length scale parameters
    #     if dg:
    #         k = 0
    #     elif xeqz:
    #         k = k * sq_dist(x[:, i].T / ell[i])
    #     else:
    #         k = k * sq_dist(x[:, i].T / ell[i], z[:, i].T / ell[i])
    # elif i == d:  # magnitude parameter
    #     k *= 2
    # else:
    #     raise NotImplementedError


if __name__ == '__main__':
    xx = np.array([[1., 2.], [3., 4.], [5., 6.]]).T
    yy = np.array([[5., 6.], [1., 2.], [3., 4.]]).T * -1
    hyp = np.array([1.1, 2.2, 3.3, 4.4])
    print(cov_se_ard(hyp, xx, yy))
    print(cov_se_ard(hyp, yy))