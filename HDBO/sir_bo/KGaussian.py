import numpy as np


def KGaussian(gamma, trainX, tildeX=None):
    """Building kernel data matrix, full or reduced.
    Args:
        trainX: full data set
        tildeX: can be full or reduced set
        gamma: width parameter; kernel value: exp(-gamma(trainX_i - trainX_j)**2)
    Returns:
        kerX: kernel data using Gaussian kernel
    """
    if tildeX is None:  # square full kernel
        aa = (trainX ** 2).sum(axis=-1, keepdims=True)
        kerX = np.exp((-aa - aa.T + 2 * trainX @ trainX.T) * gamma)
    else:
        aa = (trainX ** 2).sum(axis=-1, keepdims=True)  # (n1, 1)
        tildeAA = (tildeX ** 2).sum(axis=-1, keepdims=True)  # (n2, 1)
        kerX = np.exp((-aa - tildeAA.T + 2 * trainX @ tildeX.T) * gamma)
    return kerX


if __name__ == '__main__':
    print(KGaussian(0.1, np.arange(1, 5).reshape((2, 2)),))
    print(KGaussian(0.1, np.arange(1, 5).reshape((2, 2)), np.arange(2).reshape((1, 2))))