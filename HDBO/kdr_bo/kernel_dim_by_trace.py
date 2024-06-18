# import numpy as np
# from numpy.linalg import svd, inv, norm
import torch
from torch import Tensor
from torch.linalg import svd, solve, norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
TOL = 1e-9  # tolerance for stopping the optimization

def compute_gram_matrix(samples: Tensor):
    num = samples.shape[0]

    ab = samples @ samples.T  # shape(N,N)
    aa = torch.diag(ab)
    d = torch.repeat_interleave(aa.unsqueeze(-1), num, dim=1)  # shape(n,n)
    return d + d.T - 2 * ab


def contrast_func(
        X: Tensor, kernel_yo: Tensor, B: Tensor, Q: Tensor,
        sigma_z2: float, eps: float):

    num = X.shape[0]

    # Gram matrix of y
    # kernel_yo = (kernel_yo+kernel_yo.T)/2  # symmetrization

    # initial value of the objective function
    Z = X @ B  # shape(N, d)

    # Gram matrix of z
    zz = compute_gram_matrix(Z)
    # assert torch.all(zz >= 0).item()
    gram_z = torch.exp(-zz / sigma_z2)
    kernel_z = Q @ gram_z @ Q  # Kz(i,j)=Q*exp(-||z(i)-z(j)||^2)*Q
    # kernel_z = (kernel_z + kernel_z.T)/2  # symmetrization

    I = torch.eye(num, dtype=dtype, device=device)
    kernel_yzi = solve((kernel_z + eps * num * I).T, kernel_yo.T).T  # kernel_y0 @ inv(kernel_z)
    return torch.trace(kernel_yzi)


def kernel_dim_by_trace(X: Tensor, Y: Tensor, d: int, max_loop: int, sigma_x: float, sigma_y: float, eps, eta, anl):
    """ This program minimizes the objective function
            trace[Ky(Kz(b)+eps*I)^{-1}]
        by the gradient descend method.
    Args:
        X: explanatory variables (input data)
        Y: response variable (teaching data)
        d: low dimension
        max_loop: maximum number os iteration
        sigma_x:
        sigma_y:
        eps:
        eta:
        anl:
    Returns:
        B: orthonormal column vectors (Dxd)
    """

    num, ndims = X.shape  # n:data size, m: dim of x.
    assert d < ndims
    B = torch.randn(ndims, d, dtype=dtype, device=device)  # Random initialization of projectio matrix

    # 正交化 B.T @ B = I
    B, _, _ = svd(B, full_matrices=False)  # shape(D,d)

    # orthogonal projector to the orthogonal complement to <1>
    unit = torch.ones(num, num, dtype=dtype, device=device)
    I = torch.eye(num, dtype=dtype, device=device)
    centering_matrix = I - unit/num

    # Gram matrix of y
    sigma_y2 = 2 * sigma_y ** 2
    yy = compute_gram_matrix(Y)
    # from sklearn.metrics import pairwise_distances
    # yy2 = pairwise_distances(Y)
    # assert np.allclose(yy.sqrt().numpy(), yy2)
    # assert torch.all(yy >= 0).item()

    # main loop for minimization
    # xx_g = np.zeros((num*ndims, num))
    # for i in range(ndims):
    #     xi = np.repeat(X[:, i:i+1], num, axis=1)  # 第i维变量
    #     xx_g[i*num: (i+1)*num, :] = xi-xi.T

    ssz2 = 2 * sigma_x ** 2
    ssy2 = 2 * sigma_y ** 2

    for h in np.arange(1, max_loop+1):
        sigma_z2 = ssz2 + (anl-1)*ssz2*(max_loop-h)/max_loop
        sigma_y2 = ssy2 + (anl-1)*ssy2*(max_loop-h)/max_loop
        kernel_y = centering_matrix @ torch.exp(-yy/sigma_y2) @ centering_matrix

        B = B.requires_grad_(True)
        loss = contrast_func(X, kernel_y, B, centering_matrix, sigma_z2, eps)
        dB = torch.autograd.grad(loss, B)[0]
        B = B.requires_grad_(False)

        normdB = norm(dB)
        if (normdB < TOL).item():
            break
        dB = dB / normdB

        # dB2 = compute_gradient_4_B(X.numpy(), yy.numpy(), B.detach().numpy(), xx_g, sigma_z2, sigma_y2, eps)
        # dB2 = dB2 / np.linalg.norm(dB2)
        # assert np.allclose(dB2, dB.detach().numpy())

        # Line search
        B = kdr_line(X, kernel_y, sigma_z2, B, dB, centering_matrix, eta, eps)
        B, _, _ = svd(B, full_matrices=False)
    return B


def kdr_line(X, kernel_y, sigma_z2, B, dB,
             Q, eta, eps):
    """
    Args:
        x:
        kernel_y:
        igma_z2:
        B: projection matrix
        dB: gradient w.r.t B
        Q: centering matrix
        eta: range of golden ratio search
        eps:
    Returns:
    """
    num = X.shape[0]
    I = torch.eye(num, dtype=dtype, device=device)

    def kdr1dim(s):
        tmp_B = B - torch.tensor(s, dtype=dtype, device=device) * dB
        tmp_B, _, _ = svd(tmp_B, full_matrices=False)
        Z = X @ tmp_B
        gram_z = compute_gram_matrix(Z)
        kernel_z = Q @ torch.exp(-gram_z/sigma_z2) @ Q  # centering
        kernel_yzi = solve((kernel_z + eps * num * I).T, kernel_y.T).T  # kernel_y0 @ inv(kernel_z)
        return torch.trace(kernel_yzi).item()

    from scipy.optimize import fminbound
    s_opt = fminbound(kdr1dim, x1=0, x2=eta, maxfun=30, disp=0)

    return B - s_opt * dB


import numpy as np
from numpy import ndarray

def compute_gradient_4_B(
        X: ndarray, yy: ndarray, B: ndarray, xx_g: ndarray,
        sigma_z2: float, sigma_y2: float, eps: float):

    d = B.shape[1]
    num, ndims = X.shape
    unit = np.ones((num, num))
    I = np.eye(num)

    Z = X @ B  # shape(n,k)
    bb = np.sum(Z**2, axis=1, keepdims=True)  # shape(n,1)
    ab = Z @ Z.T  # shape(n,n)
    kernel_zw = np.abs(np.repeat(bb, num, axis=1) + np.repeat(bb.T, num, axis=0) - 2*ab)  # shape(n,n)
    kernel_zw = np.exp(- kernel_zw / sigma_z2)  # Gram_z
    m_kernel = np.mean(kernel_zw, axis=1, keepdims=True)  # shape(n,1)
    r_kernel = np.repeat(m_kernel, num, axis=1)  # shape(n,n)
    kernel_z = kernel_zw - r_kernel - r_kernel.T + np.mean(m_kernel) * unit  # centering Gram_zc
    kernel_zi = np.linalg.inv(kernel_z + eps * num * I)

    kernel_y = np.exp(-yy / sigma_y2)
    m_kernel = np.mean(kernel_y, axis=1, keepdims=True)
    r_kernel = np.repeat(m_kernel, num, axis=1)
    kernel_y = kernel_y - r_kernel - r_kernel.T + np.mean(m_kernel) * unit  # centering
    kernel_yzi = kernel_y @ kernel_zi

    zz_g = np.zeros((num * d, num))
    for i in range(d):
        zi = np.repeat(Z[:, i:i+1], num, axis=1)
        zz_g[i*num:(i+1)*num, :] = zi-zi.T

    dB = np.empty((ndims, d))
    for i in range(ndims):
        xx = xx_g[i*num:(i+1)*num, :]  # shape(n,n)
        for j in range(d):
            # tt = xx * zz_g[j*num:(j+1)*num, :] * kernel_zw  # shape(n,n)
            tt = xx * zz_g[j*num:(j+1)*num, :] * kernel_zw * (2./sigma_z2)  # shape(n,n)
            m_kernel = np.mean(tt, axis=1, keepdims=True)  # shape(n,1)
            r_kernel = np.repeat(m_kernel, num, axis=1)
            d_kernel_b = tt - r_kernel - r_kernel.T + np.mean(m_kernel) * unit  # centering
            dB[i, j] = np.trace(kernel_zi @ kernel_yzi @ d_kernel_b)
    return dB

if __name__ == '__main__':
    MAX_LOOP = 50  # number of iterations in KDR method
    EPS = 1e-4  # regularization parameter for matrix inversion
    ANL = 4  # maximum value for anealing
    ETA = 10.  # range of golden ratio search

    d = 2
    # x=np.arange(1, 7).reshape((2, 3)),
    # y=np.array([[10.], [11.]])
    x = torch.rand(5, 4, dtype=dtype, device=device)
    y = torch.randn(5, 1, dtype=dtype, device=device)
    num, ndims = x.shape  # number of data, dim of x

    x = (x - torch.mean(x, dim=0, keepdim=True))/torch.std(x, dim=0, keepdim=True, unbiased=True)  # standardization of x

    # Gaussian kernels are used. Deviation parameter are set by the median of mutual distance.
    # In the aneaning, sigma changes to 2*median to 0.5*median
    sigma_x = 0.5
    sigma_y = 0.5
    # kdr optimization. Steepest descent with line search
    b = kernel_dim_by_trace(x, y, d, MAX_LOOP, sigma_x * np.sqrt(d/ndims), sigma_y, EPS, ETA, ANL)
