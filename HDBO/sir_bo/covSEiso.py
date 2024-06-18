import numpy as np


def cov_se_iso(hyp, x, z, i):
    """
    Squared Exponential covariance function with isotropic distance measure.
    The covariance function is parameterized as:
        k(x^p, x^q) = sf^2 * exp(-(x^p - x^q).T @ inv(P) @ (x^p - x^q) / 2)
    where the P matrix is ell^2 times the unit matrix and sf^2 is the signal variance.
    The hyper-parameters are:
        hyp = [ log(ell), log(sf) ]
    For more help on design of covariance functions, try "help covFunctions".

    :param hyp:
    :param x:
    :param z:
    :param i:
    :return:
    """

    