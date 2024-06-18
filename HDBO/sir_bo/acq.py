import numpy as np
from .log_expected_imp import log_expected_improvement
from .KGaussian import KGaussian

def acq(model: dict, xx: np.ndarray, acq_type='ucb'):
    """
    Args:
        model: Hyper-parameters.
        xx: (num, high_dims). Data matrix.
        project_mat: (high_dims, dims). Projection matrix.
    Returns:
    """

    # x translate based on A original input x.T
    train_X = xx[np.newaxis, :] @ model.project_mat
    mu, var = model.mean_var(train_X)
    sigma = np.sqrt(max(var, 0))

    if acq_type == 'ucb':
        coefficient = np.sqrt(2 * np.log(model['n'] ** (model['d']/2+2) * np.pi**2 / 0.3))
        return (mu + coefficient * sigma) * -1  # Flip the objective for maximization
    elif acq_type == 'ei':
        # Flip the objective for maximization
        return log_expected_improvement(model.min_val, mu.ravel() + 1e-4, sigma.ravel()) * -1
    else:
        return NotImplementedError


def kerAcq(model: dict, xx: np.ndarray, acq_type='ucb'):
    """
    Args:
        model: Hyper-parameters.
        xx: (num, high_dims). Data matrix.
        project_mat: (high_dims, dims). Projection matrix.
    Returns:
    """
    kerX = KGaussian(model.gamma, xx[np.newaxis, :], model.centerX)
    # x translate based on A original input x.T
    train_X = kerX @ model.project_mat
    mu, var = model.mean_var(train_X)
    # assert var > 0
    sigma = np.sqrt(max(var, 0))

    if acq_type == 'ucb':
        coefficient = np.sqrt(2 * np.log(model['n'] ** (model['d']/2+2) * np.pi**2 / 0.3))
        return (mu + coefficient * sigma) * -1  # Flip the objective for maximization
    elif acq_type == 'ei':
        # Flip the objective for maximization
        return log_expected_improvement(model.min_val, mu.ravel() + 1e-4, sigma.ravel()) * -1
    else:
        return NotImplementedError
