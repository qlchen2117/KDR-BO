import numpy as np
import numpy.linalg as LA


def sir_dir(x: np.ndarray, y: np.ndarray, num_of_slice: int, dim: int):
    """
    Args:
        x: (num, high_dim) data points
        y: (num,) evaluation
        num_of_slice
        dim: embedding dim
    Returns: 
        numpy array (num, dim)
    """
    assert y.ndim == 1
    n, p = x.shape
    indices = np.argsort(y)
    x = x[indices]
    x_mean = np.mean(x, axis=0)  # (high_dim,)

    # extract centered and weighted slice means for regression
    size_of_slice = n // num_of_slice
    m = n % num_of_slice  # remained data points
    pointer0, pointer1 = 0, 0
    sample_mean_c = np.zeros((num_of_slice, p))
    for i in range(num_of_slice):
        count = size_of_slice + (i < m)  # add remained points to the first m slices
        pointer1 += count
        sample_mean_c[i] = (np.mean(x[pointer0: pointer1, :], axis=0) - x_mean) * np.sqrt(count/n)
        # i-th slice mean, centered
        pointer0 = pointer1

    # solve the following generalized eigenvalue problem
    # Cov(HX)*V = lambda * Cov(X) * V
    cov_x = x.T @ x / n - x_mean[:, np.newaxis] @ x_mean[np.newaxis, :]  # (high_dim, high_dim)
    cov_xi_w = LA.solve((cov_x + 1e-10 * np.eye(p)), sample_mean_c.T)  # compute inv(cov_x) @ W
    wt_cov_xi_w = sample_mean_c @ cov_xi_w  # W.T @ inv(cov_x) @ W
    wt_cov_xi_w = (wt_cov_xi_w + wt_cov_xi_w.T) / 2  # ensure the matrix is symmetrix
    eigvals, eigvects = LA.eigh(wt_cov_xi_w)  # extract v via solving W.T @ inv(cov_x) @ W v = v diag(w)

    eigvals, eigvects = np.maximum(eigvals[::-1][:dim], np.finfo(np.float64).eps), eigvects[:, ::-1][:, :dim]  # sort in descend order
    return cov_xi_w @ (eigvects / np.sqrt(eigvals))


# if __name__ == '__main__':
#     X = np.random.randn(100, 5)
#     Y = X @ (np.array([1, 1, 1, 1, 0])[:, np.newaxis]) + np.random.randn(1)
#     ans = sir_dir(X, Y.squeeze(), num_of_slice=5)
#     print(ans)

if __name__ == '__main__':
    ans = sir_dir(x=np.arange(1, 22).astype(float).reshape(7, 3), y=np.arange(1, 8).astype(int)[::-1], num_of_slice=3, dim=2)
    print(ans)
