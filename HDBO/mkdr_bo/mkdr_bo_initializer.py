from scipy.stats import qmc
import numpy as np
import torch
import gpytorch


def gen(model, batchgp, n, dim, dtype, device):
    """Sample in a low-dimensional linear embedding, to initialize ALEBO.

    Generates points on a linear subspace of [-1, 1]^D by generating points in
    [-b, b]^D, projecting them down with mkdr, and then projecting them
    back up with MOGP. Thus points thus all lie in a linear
    subspace defined by mkdr. Points whose up-projection falls outside of [-1, 1]^D
    are thrown out, via rejection sampling.

    Args:
        mkdr: projection down.
        nsamp: Number of samples to use for rejection sampling.
    """

    # 在[0,1]^D上uniform采样10000个点
    Xrand = np.empty((0, dim))
    iter = 0
    while True:
        X01 = qmc.Sobol(d=dim, scramble=True).random_base2(m=10)
        X_b = 2 * X01 - 1  # map to [-1, 1]^D
        low_bound = model.gkernel(model.dmap.X_fit_, model.dmap.X_fit_).sum(axis=1).min() - 1
        row_sum = model.gkernel(model.dmap.X_fit_, X_b).sum(axis=1)
        indice = row_sum.A.ravel() >= low_bound
        X_b = X_b[indice]
        Zrand = torch.from_numpy(model.transform(X_b)).to(dtype=dtype, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn = batchgp.posterior(Zrand.unsqueeze(1))
            mean, stddev = mvn.mean.squeeze(1), mvn.stddev.squeeze(1)
        X_b = X_b[(mean - stddev >= -1.0).all(axis=1) & (mean + stddev <= 1.0).all(axis=1)]
        Xrand = np.vstack((Xrand, X_b))
        iter += 1
        if Xrand.shape[0] >= n:
            return iter, Xrand
    raise NotImplementedError
