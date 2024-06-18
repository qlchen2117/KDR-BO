import numpy as np
from numpy.linalg import svd, inv, norm
from .kdr_line import kdr_line
from .kernel_derivative import kernel_derivative


def kernel_dim_red_trace(x, y, b, max_loop, sigma_x, sigma_y, eps, eta, anl):
    """ This program minimizes the objective function
            trace[Ky(Kz(b)+eps*I)^{-1}]
        by the gradient descend method.
    Args:
        x: explanatory variables (input data)
        y: response variable (teaching data)
        b: dimension
        max_loop: maximum number os iteration
        sigma_x:
        sigma_y:
        eps:
        eta:
        anl:
    Returns:
        b: orthonormal column vectors (MxK)
    """
    tol = 1e-9  # tolerance for stopping the optimization
    init_derivative = False  # True: initialization by derivative method. False: random

    num, ndims = x.shape  # n:data size, m: dim of x.
    length = y.shape[1]
    if type(b) is not np.ndarray:
        k = b
        if init_derivative:
            b, t = kernel_derivative(x, y, k, np.sqrt(anl)*sigma_x, sigma_y, eps)
        else:
            b = np.random.randn(ndims, k)
    else:
        k = b.shape[-1]
    if k >= ndims:
        raise Exception("Dimension of the effective subspace should be smaller than the dimension of X")

    # 正交化 b.T @ b = I
    b, _, _ = svd(b, full_matrices=False)  # shape(m,k)

    # orthogonal projector to the orthogonal complement to <1>
    unit = np.ones((num, num))
    identity = np.eye(num)
    centering_matrix = identity - unit/num

    # Gram matrix of y
    sigma_y2 = 2 * sigma_y ** 2
    aa = np.sum(y**2, axis=1, keepdims=True)  # shape(n,1)
    ab = y @ y.T  # shape(n,n)
    d = np.repeat(aa, num, axis=1)  # shape(n,n)
    yy = np.maximum(d+d.T-2*ab, 0)
    gram_y = np.exp(-yy/sigma_y2)
    kernel_yo = centering_matrix @ gram_y @ centering_matrix
    kernel_yo = (kernel_yo+kernel_yo.T)/2  # Ky(i,j)=Q@exp(-||yi-yj||^2/sigma)@Q

    # initial value of the objective function
    z = x @ b  # shape(n, k)
    nz = z/np.sqrt(2)/sigma_x
    aa = np.sum(nz**2, axis=1, keepdims=True)  # shape(n,1)
    ab = nz @ nz.T
    d = np.repeat(aa, num, axis=1)  # shape(n,n)
    zz = np.maximum(d+d.T-2*ab, 0)
    gram_z = np.exp(-zz)
    kernel_z = centering_matrix @ gram_z @ centering_matrix
    kernel_z = (kernel_z + kernel_z.T)/2  # Kz(i,j)=Q*exp(-||z(i)-z(j)||^2)*Q

    mz = inv(kernel_z + eps * num * identity)
    trace = np.sum(kernel_yo*mz)

    # main loop for minimization
    xx_g = np.zeros((num*ndims, num))
    for i in range(ndims):
        xi = np.repeat(x[:, i:i+1], num, axis=1)  # 第i维变量
        xx_g[i*num: (i+1)*num, :] = xi-xi.T

    ssz2 = 2*sigma_x**2
    ssy2 = 2*sigma_y**2

    for h in np.arange(1, max_loop+1):
        sigma_z2 = ssz2 + (anl-1)*ssz2*(max_loop-h)/max_loop
        sigma_y2 = ssy2 + (anl-1)*ssy2*(max_loop-h)/max_loop

        z = x @ b  # shape(n,k)
        bb = np.sum(z**2, axis=1, keepdims=True)  # shape(n,1)
        ab = z @ z.T  # shape(n,n)
        kernel_zw = np.abs(np.repeat(bb, num, axis=1) + np.repeat(bb.T, num, axis=0) - 2*ab)  # shape(n,n)
        kernel_zw = np.exp(-kernel_zw/sigma_z2)  # Gram_z
        m_kernel = np.mean(kernel_zw, axis=1, keepdims=True)  # shape(n,1)
        r_kernel = np.repeat(m_kernel, num, axis=1)  # shape(n,n)
        kernel_z = kernel_zw - r_kernel - r_kernel.T + np.mean(m_kernel) * unit  # centering Gram_zc
        kernel_zi = inv(kernel_z+eps*num*identity)

        kernel_y = np.exp(-yy/sigma_y2)
        m_kernel = np.mean(kernel_y, axis=1, keepdims=True)
        r_kernel = np.repeat(m_kernel, num, axis=1)
        kernel_y = kernel_y - r_kernel - r_kernel.T + np.mean(m_kernel)*unit  # centering
        kernel_yzi = kernel_y @ kernel_zi

        zz_g = np.zeros((num*k, num))
        for i in range(k):
            zi = np.repeat(z[:, i:i+1], num, axis=1)
            zz_g[i*num:(i+1)*num, :] = zi-zi.T

        db = np.empty((ndims, k))
        for i in range(ndims):
            xx = xx_g[i*num:(i+1)*num, :]  # shape(n,n)
            for j in range(k):
                tt = xx * zz_g[j*num:(j+1)*num, :] * kernel_zw  # shape(n,n)
                m_kernel = np.mean(tt, axis=1, keepdims=True)  # shape(n,1)
                r_kernel = np.repeat(m_kernel, num, axis=1)
                d_kernel_b = tt - r_kernel - r_kernel.T + np.mean(m_kernel)*unit  # centering
                db[i, j] = np.trace(kernel_zi @ kernel_yzi @ d_kernel_b)
        if norm(db) < tol:
            break
        # Line search
        nm = norm(db)
        b = kdr_line(x, kernel_y, sigma_z2, b, db/nm, eta, eps)
        b, _, _ = svd(b, full_matrices=False)
    return b

