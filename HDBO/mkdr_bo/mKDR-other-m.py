import numpy as np
import torch
from torch.quasirandom import SobolEngine
from matplotlib import pyplot as plt
import math
import numpy.linalg as LA
from scipy.optimize import minimize
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from typing import Callable


def line_search(f_and_df: Callable, x0: np.ndarray, f0: np.ndarray, df0: np.ndarray, direction: np.ndarray, lr):
    c1, c2 = 1e-4, 0.999
    dpsi0 = np.dot(df0, direction)
    psi0 = f0

    def zoom(alpha1, alpha2, psi1):
        while np.abs(alpha1 - alpha2) > 1e-9:
            alpha_ = (alpha1 + alpha2) / 2
            x_ = x0 + alpha_ * direction
            psi_, df_, x_ = f_and_df(x_)
            dpsi_ = np.dot(df_, direction)
            if psi_ > psi0 + c1 * alpha_ * dpsi0 or psi_ >= psi1:
                alpha2 = alpha_
            else:
                if np.abs(dpsi_) <= -1 * c2 * dpsi0:
                    return x_, psi_, df_
                if dpsi_ * (alpha2 - alpha1) >= 0:
                    alpha2 = alpha1
                alpha1, psi1 = alpha_, psi_

        raise NotImplementedError("Fail in zoom")

    alpha, alpha_ = 0, lr
    psi = psi0
    for t in range(30):
        x_ = x0 + alpha_ * direction
        psi_, df_, x_ = f_and_df(x_)
        dpsi_ = np.dot(df_, direction)
        # violates the sufficient decrease condition
        if psi_ > psi0 + c1 * alpha_ * dpsi0 or (psi_ >= psi and t > 1):
            return zoom(alpha, alpha_, psi)
        if np.abs(dpsi_) <= -1 * c2 * dpsi0:  # wolfe condition 2
            return x_, psi_, df_
        if dpsi_ >= 0:
            return zoom(alpha_, alpha, psi_)
        alpha, psi = alpha_, psi_
        alpha_ = (alpha_ + 1) / 2
    raise NotImplementedError("Violates wolfe condition 2")


DIM = 10
n_eigenpairs = 10

embeddings = {
    # "Isomap": Isomap(n_components=50),
    # "Standard-LLE": LocallyLinearEmbedding(
    #     n_components=10, method="standard"
    # ),
    # "Modified-LLE": LocallyLinearEmbedding(
    #     n_neighbors=10, n_components=10, method="modified"
    # ),
    # "Hessian-LLE": LocallyLinearEmbedding(
    #     n_neighbors=10*13//2+1, n_components=10, method="hessian"
    # ),
    # "LTSA-LLE": LocallyLinearEmbedding(
    #     n_neighbors=10, n_components=10, method="ltsa"
    # ),
    # "MDS": MDS(
    #     n_components=50, n_init=1, max_iter=120, n_jobs=2, normalized_stress="auto"
    # ),
    # "Spectral": SpectralEmbedding(
    #     n_components=50, eigen_solver="arpack"
    # ),
    "t-SNE": TSNE(
        n_components=10,
        method="exact",
    ),
}


class MKDR_isomap:
    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def fit_transform(self, X: np.ndarray, Y: np.ndarray, d: int):
        eps = 1e-4  # regularization parameter for matrix inversion
        num = X.shape[0]

        U = (self.transformer.fit_transform(X)).T

        kernel_y = Y @ Y.T + num * eps * np.eye(num)
        m_kernel = np.mean(kernel_y, axis=1, keepdims=True)  # shape(n,1)
        r_kernel = np.repeat(m_kernel, num, axis=1)  # shape(n,n)
        kyc = kernel_y - r_kernel - r_kernel.T + np.mean(m_kernel)  # centering Q @ ky @ Q

        # def f_np_wrapper(omega: np.ndarray):
        #     """Given a torch callable, compute value + grad given a numpy array."""
        #     omega = omega.reshape((n_eigenpairs, n_eigenpairs))
        #     # Project symmetric Omega to the positive semidefinite cone
        #     omega = (omega + omega.T) / 2  # make sure Omega is symmetric
        #     e_vals, e_vecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
        #     e_vals = np.maximum(e_vals, 0)
        #     omega = (e_vecs * e_vals) @ e_vecs.T

        #     # scale Omega to have unit trace
        #     omega /= np.trace(omega)

        #     # the objective function
        #     kx = U.T @ omega @ U + num * eps * np.eye(num)
        #     objective = np.trace(LA.solve(kx.T, kyc.T).T)  # kyc @ LA.inv(kx)

        #     # compute gradient
        #     kxi_ut = LA.solve(kx, U.T)
        #     gradf = -1 * kxi_ut.T @ kyc @ kxi_ut  # -1 * U @ kxi @ kernel_yc @ kxi @ U.T, kxi = LA.inv(kx)

        #     fval = objective.item()
        #     return fval, gradf.ravel()

        # res = minimize(
        #     fun=f_np_wrapper,
        #     x0=np.eye(n_eigenpairs).ravel(),
        #     method='L-BFGS-B',
        #     jac=True,
        # )
        # omega = res.x.reshape((n_eigenpairs, n_eigenpairs))


        def f_and_df(omega: np.ndarray):
            """compute value + grad given a numpy array."""
            omega = omega.reshape((n_eigenpairs, n_eigenpairs))
            # Project symmetric Omega to the positive semidefinite cone
            omega = (omega + omega.T) / 2  # make sure Omega is symmetric
            e_vals, e_vecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
            e_vals = np.maximum(e_vals, 0)
            omega = (e_vecs * e_vals) @ e_vecs.T

            # scale Omega to have unit trace
            omega /= np.trace(omega)

            # the objective function
            kx = U.T @ omega @ U + num * eps * np.eye(num)
            objective = np.trace(LA.solve(kx.T, kyc.T))  # kyc @ LA.inv(kx)

            # compute gradient
            kxi_ut = LA.solve(kx, U.T)
            gradf = -1 * kxi_ut.T @ kyc @ kxi_ut  # -1 * U @ kxi @ kernel_yc @ kxi @ U.T, kxi = LA.inv(kx)

            fval = objective.item()
            return fval, gradf.ravel(), omega.ravel()
            # return fval, gradf.ravel()

        # ======================= line search II ====================
        x0 = np.eye(n_eigenpairs).ravel()
        f0, df0, x0 = f_and_df(x0)
        iter = 1
        try:
            while LA.norm(df0) > 1e-5:
                x1, f1, df1 = line_search(f_and_df, x0, f0, df0, -1 * df0 / LA.norm(df0), 1/iter)
                x0, f0, df0 = x1, f1, df1
                print(f"iter{iter}, f={f1}, df-norm={LA.norm(df1)}")
                iter += 1
        except NotImplementedError as e:
            print(e)
        omega = x0.reshape((n_eigenpairs, n_eigenpairs))

        # compute Phi as the square root of Omega
        omega = (omega + omega.T) / 2  # make sure Omega is symmetric
        omega /= np.trace(omega)
        eigvals, eigvecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
        eigvals = np.maximum(eigvals, 0)
        eigvals_sq, eigvecs = np.sqrt(eigvals[-d:]), eigvecs[:, -d:]
        eigvals_sq /= eigvals_sq.sum()  # as weight
        eigvecs = eigvecs / LA.norm(eigvecs, axis=0)  # normalize

        # self.isomap_embedding = isomap_embedding
        self.omega_eigvals_sq = eigvals_sq
        self.omega_eigvecs = eigvecs
    
        return U.T @ (eigvecs * eigvals_sq)  #  U.T @ eigvecs @ np.diag(eigvals)



def random_sample_angles(n_init=1000):
    theta = SobolEngine(dimension=2, scramble=True).draw(n_init).to(dtype=torch.double)
    theta *= 2 * torch.pi  # scale to [0, 2pi]^2
    x1 = (2 + torch.cos(theta[:, 0])) * torch.cos(theta[:, 1])
    x2 = (2 + torch.cos(theta[:, 0])) * torch.sin(theta[:, 1])
    x3 = torch.sin(theta[:, 0])
    xx = torch.cat((x1.view(-1, 1), x2.view(-1, 1), x3.view(-1, 1)), dim=1)
    y = torch.sigmoid( -17 *
        (torch.sqrt(torch.sum((theta - torch.pi) ** 2, dim=1)) - 0.6 * torch.pi)
    )
    # y = torch.sqrt(torch.sum((theta - torch.pi) ** 2, dim=1))
    return xx, y.view(-1, 1), theta


def plot_embedding(X, Y, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.scatter(X[:, -1], Y[:, 0])
    plt.savefig(f"testMKDR-{title}.png")


if __name__ == '__main__':
    # N_EIG = 50
    # plot_torus()
    X, Y, theta = random_sample_angles()
    X = torch.cat((X, torch.zeros(X.shape[0], DIM-X.shape[1])), dim=1)  # embed the torus in R^10
    train_Y = (Y - Y.mean()) / Y.std()
    # train_X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    # train_X = torch.where(torch.isnan(train_X), 0, train_X)
    projections = {}
    for name, transformer in embeddings.items():
        mkdr = MKDR_isomap(transformer)
        projections[name] = mkdr.fit_transform(X.numpy(), train_Y.numpy(), d=1)
        plot_embedding(projections[name], Y, title=name)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    # ax2.scatter(theta[:, 0], theta[:, 1], Y[:, 0])
