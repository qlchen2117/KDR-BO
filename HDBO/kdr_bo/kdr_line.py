import numpy as np
from numpy.linalg import svd, inv
from scipy.optimize import fminbound


def kdr_line(x, kernel_y, sigma_z2, b, db, eta, eps):
    """
    Args:
        x:
        kernel_y:
        igma_z2:
        b:
        db:
        eta:
        eps:
    Returns:
    """
    num = x.shape[0]
    k = b.shape[1]
    unit = np.ones((num, num))

    def kdr1dim(s):
        tmp_b = b - s * db
        tmp_b, _, _ = svd(tmp_b, full_matrices=False)
        z = x @ tmp_b
        ab = z @ z.T
        diag_ab = np.diag(ab)
        d = np.repeat(diag_ab[:, np.newaxis], num, axis=1)
        kernel_z = np.exp((-d-d.T+2*ab)/sigma_z2)
        m_kernel = np.mean(kernel_z, axis=1, keepdims=True)  # shape(n,1)
        r_kernel = np.repeat(m_kernel, num, axis=1)
        kernel_z = kernel_z - r_kernel - r_kernel.T + np.mean(m_kernel)*unit  # centering
        mz = inv(kernel_z + num*eps*np.eye(num))
        return np.sum(kernel_y * mz)

    s_opt = fminbound(kdr1dim, x1=0, x2=eta)
    return b - s_opt * db
