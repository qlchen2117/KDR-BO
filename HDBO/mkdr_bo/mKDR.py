import logging
import numpy as np
import numpy.linalg as LA
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from typing import Callable
import logging
# from scipy.optimize import minimize, Bounds, fminbound
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import pairwise_distances

def gram_schmidt(A):
    n = A.shape[1]

    A[:, 0] = A[:, 0] / LA.norm(A[:, 0])

    for i in range(1, n):
        Ai = A[:, i]
        for j in range(0, i):
            Aj = A[:, j]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        A[:, i] = Ai / LA.norm(Ai)
    return A


def median_dist(x):
    n = x.shape[0]  # number of data
    ab = x @ x.T  # (xi.T @ xj)_ij
    aa = np.diag(ab)
    dx = aa[:, np.newaxis] + aa[np.newaxis, :] - 2*ab # shape(n,n)
    dx -= np.diag(np.diag(dx))  # dx = xi.T @ xi + xj.T @ xj - 2*xi.T @ xj
    dx = dx[np.nonzero(dx)]
    return np.sqrt(np.median(dx))


def compute_gram_matrix(samples):
    num = samples.shape[0]

    ab = samples @ samples.T  # shape(N,N)
    aa = np.diag(ab)
    d = np.repeat(aa[:, np.newaxis], num, axis=1)  # shape(n,n)
    return d + d.T - 2 * ab


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
        alpha_ = (alpha_ + 10) / 2
    raise NotImplementedError("Violates wolfe condition 2")


class MKDR:
    def fit_transform(self, X: np.ndarray, Y: np.ndarray, n_eigenpairs: int, d: int):
        eps = 1e-4  # regularization parameter for matrix inversion
        num = X.shape[0]
        # scaler = StandardScaler().fit(X)
        # X = scaler.transform(X)

        # compute eigenvectors via diffusion maps
        X_pcm = pfold.PCManifold(X)
        X_pcm.optimize_parameters()
        gkernel = pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off))
        dmap = dfold.DiffusionMaps(
            gkernel,
            n_eigenpairs=n_eigenpairs,
            # is_stochastic=False
        ).fit(X_pcm)

        indice = np.nonzero(dmap.eigenvalues_ > 1e-14)[0][1:]
        dmap = dmap.set_target_coords(indice.astype(int))
        evecs, evals = dmap._select_eigenpairs_target_coords()
        # dmap_weights =  evals / evals.sum()
        U = evecs.T  # U = (dmap_weights * evecs).T  # shape(n_eigenpairs, N)
        n_eigenpairs = len(indice)
        # U = dmap.eigenvectors_[:, 1:].T

        kernel_y = Y @ Y.T + num * eps * np.eye(num)
        m_kernel = np.mean(kernel_y, axis=1, keepdims=True)  # shape(n,1)
        r_kernel = np.repeat(m_kernel, num, axis=1)  # shape(n,n)
        kyc = kernel_y - r_kernel - r_kernel.T + np.mean(m_kernel)  # centering Q @ ky @ Q

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

        # ======================= line search ====================
        x0 = np.eye(n_eigenpairs).ravel()
        f0, df0, x0 = f_and_df(x0)
        iter = 1
        try:
            while LA.norm(df0) > 1e-5:
                x1, f1, df1 = line_search(f_and_df, x0, f0, df0, -1 * df0 / LA.norm(df0), 1/iter)
                x0, f0, df0 = x1, f1, df1
                logging.debug(f"iter{iter}, f={f1}, df-norm={LA.norm(df1)}")
                iter += 1
        except NotImplementedError as e:
            pass
            # logging.warning(e)

        omega = x0.reshape((n_eigenpairs, n_eigenpairs))

        # compute Phi as the square root of Omega
        omega = (omega + omega.T) / 2  # make sure Omega is symmetric
        eigvals, eigvecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
        eigvals = np.maximum(eigvals, 0)
        eigvals_sq, eigvecs = np.sqrt(eigvals[-d:]), eigvecs[:, -d:]  # the top d eigenvalues and eigenvectors
        eigvals_sq /= eigvals_sq.sum()  # as weight
        eigvecs = eigvecs / LA.norm(eigvecs, axis=0)  # normalize

        # self.scaler = scaler
        self.gkernel = gkernel
        self.dmap = dmap
        # self.dmap_weights = dmap_weights
        self.omega_eigvals_sq = eigvals_sq
        self.omega_eigvecs = eigvecs

        return U.T @ (eigvecs * eigvals_sq)  #  U.T @ eigvecs @ np.diag(eigvals)


    def transform(self, X_test: np.ndarray):
        # X_test = self.scaler.transform(X_test)
        X_pcm_test = pfold.PCManifold(X_test)
        X_dmap_test = self.dmap.transform(X_pcm_test)
        U = X_dmap_test.T  # U = (self.dmap_weights * X_dmap_test).T
        return U.T @ (self.omega_eigvecs * self.omega_eigvals_sq)



"""
# ======================== only Diffusion Map ====================
if __name__ == '__main__':
    N_EIG = 50
    X, Y, theta = random_sample_angles()
    X = torch.cat((X, torch.zeros(X.shape[0], DIM-X.shape[1])), dim=1)  # embed the torus in R^10
    X = X.cpu().numpy()
    # isomap = Isomap(n_neighbors=n_neighbors, n_components=n_eigenpairs)
    # projection = isomap.fit_transform(X)
    pcm = pfold.PCManifold(X)
    pcm.optimize_parameters()
    dmap = dfold.DiffusionMaps(
        pfold.GaussianKernel(
            epsilon=pcm.kernel.epsilon,
            # distance=dict(cut_off=pcm.cut_off)
        ),
        n_eigenpairs=2,
    )
    dmap = dmap.fit(pcm)
    dmap_weights = dmap.eigenvalues_ / dmap.eigenvalues_.sum()
    projection = dmap_weights * dmap.eigenvectors_
    plot_embedding(projection, Y, title="onlyDM")

        # =========================lbgsb===============================
        U, kyc = torch.from_numpy(U), torch.from_numpy(kyc)
        def f_np_wrapper(omega: np.ndarray):
            # compute value + grad given a numpy array.
            omega = torch.from_numpy(omega).requires_grad_(True).view(n_eigenpairs, n_eigenpairs)
            # Project symmetric Omega to the positive semidefinite cone
            omega = (omega + omega.T) / 2  # make sure Omega is symmetric
            e_vals, e_vecs = torch.linalg.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
            e_vals = torch.maximum(e_vals, torch.zeros_like(e_vals))
            omega = (e_vecs * e_vals) @ e_vecs.T

            # scale Omega to have unit trace
            omega /= torch.trace(omega)

            # the objective function
            kx = U.T @ omega @ U + num * eps * torch.eye(num)
            loss = torch.trace(torch.linalg.solve(kx.T, kyc.T))  # kyc @ LA.inv(kx)

            gradf = torch.autograd.grad(loss, omega)[0].contiguous().detach().cpu().numpy()

            fval = loss.item()
            # return fval, gradf.ravel(), omega.ravel()
            return fval, gradf.ravel()

        x0, df_norm = np.eye(n_eigenpairs).ravel(), 1
        while df_norm > 1e-5:
            res = minimize(f_np_wrapper, x0, method="L-BFGS-B", jac=True)
            x0 = res.x
            omega = x0.reshape((n_eigenpairs, n_eigenpairs))
            # Project symmetric Omega to the positive semidefinite cone
            omega = (omega + omega.T) / 2  # make sure Omega is symmetric
            e_vals, e_vecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
            print(f"Is all eigenvalues positive: {np.all(e_vals>=0)}")
            e_vals = np.maximum(e_vals, 0)
            omega = (e_vecs * e_vals) @ e_vecs.T

            # scale Omega to have unit trace
            omega /= np.trace(omega)
            x0 = omega.ravel()
            fval, df = f_np_wrapper(x0)
            df_norm = LA.norm(df)
            print(f"fval: {fval}, df_norm:{df_norm}")


        ========================= line search ========================
        x0 = np.eye(n_eigenpairs).ravel()
        df0, x0 = compute_grad(x0)
        direction = -1 * df0 / LA.norm(df0)

        def objective(alpha):
            x1 = x0 + alpha * direction
            dim = len(x1) // 2
            omega = x1.reshape((n_eigenpairs, n_eigenpairs))
            # Project symmetric Omega to the positive semidefinite cone
            omega = (omega + omega.T) / 2  # make sure Omega is symmetric
            e_vals, e_vecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
            e_vals = np.maximum(e_vals, 0)
            omega = (e_vecs * e_vals) @ e_vecs.T

            # scale Omega to have unit trace
            omega /= np.trace(omega)

            # the objective function
            kx = U.T @ omega @ U + num * eps * np.eye(num)
            return np.trace(LA.solve(kx.T, kyc.T))  # kyc @ LA.inv(kx)
        df_norm = 1
        while df_norm > 1e-5:
            alpha_opt = fminbound(objective, x1=0, x2=10, maxfun=30, disp=0)
            x0 = x0 + alpha_opt * direction
            f0, df0, x0 = f_and_df(x0)
            df_norm = LA.norm(df0)
            direction = -1 * df0 / df_norm
            print(f"fval: {f0}, df-norm: {df_norm}")



class MKDR:
    def fit_transform(self, X: np.ndarray, Y: np.ndarray, n_eigenpairs: int, d: int):
        tol = 1e-9  # tolerance for stopping the optimization
        eps = 1e-4  # regularization parameter for matrix inversion
        num = X.shape[0]

        # compute eigenvectors via diffusion maps
        pcm = pfold.PCManifold(X)
        pcm.optimize_parameters()
        dmap = dfold.DiffusionMaps(
            pfold.GaussianKernel(
                epsilon=pcm.kernel.epsilon,
                # distance=dict(cut_off=pcm.cut_off)
            ),
            n_eigenpairs=n_eigenpairs+1,
        )
        dmap = dmap.fit(pcm)
        # evals = dmap.eigenvalues_[1:]
        # dmap_weights =  evals / evals.sum()
        # U = (dmap_weights * dmap.eigenvectors_[:, 1:]).T  # shape(n_eigenpairs, N)

        U = dmap.eigenvectors_[:, 1:].T
        # sigma_y = median_dist(Y)
        # sigma_y2 = 2 * sigma_y ** 2
        # yy = compute_gram_matrix(Y)
        # kernel_y = np.exp(-yy/sigma_y2)
        kernel_y = Y @ Y.T + num * eps * np.eye(num)
        m_kernel = np.mean(kernel_y, axis=1, keepdims=True)  # shape(n,1)
        r_kernel = np.repeat(m_kernel, num, axis=1)  # shape(n,n)
        kyc = kernel_y - r_kernel - r_kernel.T + np.mean(m_kernel)  # centering Q @ ky @ Q

        t, objective_, omega = 1, 0, np.eye(n_eigenpairs)
        while True:
            eta = 1. / t
            # the objective function
            kx = U.T @ omega @ U + num * eps * np.eye(num)
            objective = np.trace(LA.solve(kx.T, kyc.T))  # Tr[kyc @ inv(kx)]
            if math.fabs((objective - objective_) / objective) < tol:
                break

            # compute gradient
            kxi_ut = LA.solve(kx, U.T)
            grad = -1 * kxi_ut.T @ kyc @ kxi_ut  # -1 * U @ kxi @ kernel_yc @ kxi @ U.T, kxi = LA.inv(kx)
            # gradient descent
            omega = omega - eta * grad

            # Project symmetric Omega to the positive semidefinite cone
            omega = (omega + omega.T) / 2  # make sure Omega is symmetric
            e_vals, e_vecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
            e_vals = np.maximum(e_vals, 0)
            omega = (e_vecs * e_vals) @ e_vecs.T

            # scale Omega to have unit trace
            omega /= np.trace(omega)

            t += 1
            objective_ = objective

        # compute Phi as the square root of Omega
        omega = (omega + omega.T) / 2  # make sure Omega is symmetric
        omega /= np.trace(omega)  # scale Omega to have unit trace
        eigvals, eigvecs = LA.eigh(omega)  # ascending order eigenvalues, normalized eigenvectors
        eigvals = np.maximum(eigvals, 0)
        eigvals_sq, eigvecs = np.sqrt(eigvals[::-1]), eigvecs[:, ::-1]  # decending order
        # eigvals_sq /= eigvals_sq.sum()  # as weight

        self.dmap = dmap
        self.omega_eigvals_sq = eigvals_sq
        self.omega_eigvecs = eigvecs
    
        # return U.T @ (eigvecs[:, :d] * eigvals_sq[:d])  #  U.T @ eigvecs @ np.diag(eigvals)[:, :d])
        return U.T @ eigvecs[:, :d]
        # Phi = (e_vecs.T)[-d:]  # linear map
        # return (Phi @ U).T

    def transform(self, X_test: np.ndarray, d: int):
        X_pcm_test = pfold.PCManifold(X_test)
        X_dmap_test = self.dmap.transform(X_pcm_test)
        U = (self.dmap_weights * X_dmap_test).T
        return U.T @ (self.omega_eigvecs[:, :d] * self.omega_eigvals_sq[:d])
"""