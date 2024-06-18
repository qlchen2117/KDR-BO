from typing import Callable
import torch
import time
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning, InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel

from scipy.optimize import minimize, LinearConstraint
import numpy as np
from .KDR_sample import kdr_sample

import warnings
# warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
logger.addHandler(streamHandler)

device = torch.device("cpu")
dtype = torch.double

def gen(projector, n, init_bound: int = 16):

    # 在[0,1]^D上uniform采样10000个点
    X01 = SobolEngine(dimension=projector.shape[1], scramble=True).draw(10*n).to(dtype=dtype, device=device)

    finished = False
    b = float(init_bound)
    while not finished:
        # Map to [-b, b]
        X_b = 2 * b * X01 - b  # 为什么要扩展到[-b,b]?
        # Project down to B and back up
        X = X_b @ projector.T  # x' = (B.T)^+ B.T x
        # Filter out to points in [-1, 1]^D
        X = X[(X >= -1.0).all(axis=1) & (X <= 1.0).all(axis=1)]
        if X.shape[0] >= n:
            finished = True
        else:
            b = b / 2.0  # Constrict the space
    X = X[:n, :]
    return X

def acqf_optimizer(
    acq_function,
    n: int,
    inequality_constraints,
    raw_samples: int,
    num_restarts: int,
    B: Tensor,
    projector: Tensor
):
    """
    Optimize the acquisition function for ALEBO.

    We are optimizing over a polytope within the subspace, and so begin each
    random restart of the acquisition function optimization with points that
    lie within that polytope.
    """
    dim, emb_dim = B.shape

    # 在[-1,1]^D上均匀采样1000个点
    Xrnd = gen(projector, n=raw_samples)
    Xrnd = Xrnd.unsqueeze(1)
    # 把随机点嵌入低维空间 shape(b, 1, ndims)
    Yrnd = torch.matmul(Xrnd, B)  # Project down to the embedding
    
    import gpytorch
    with gpytorch.settings.max_cholesky_size(2000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with torch.no_grad():
                alpha = acq_function(Yrnd)  # shape(b,)
        # 在b个随机点中选num_restarts个点，且y越大越可能被选中
        from botorch.optim.initializers import initialize_q_batch_nonneg
        Yinit = initialize_q_batch_nonneg(X=Yrnd, Y=alpha, n=num_restarts)

        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(Yinit)
                .view(-1, 1, emb_dim)  # shape(num_restarts x q, emb_dim)
                .contiguous()
                .requires_grad_(True)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                loss = f(X).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = torch.autograd.grad(loss, X)[0].contiguous().view(-1).cpu().numpy()
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. "
                    "This often indicates numerical issues."
                )
                if Yinit.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf

        def f(x):
            return -acq_function(x)

        Yinit = Yinit.squeeze(1)  # (b, emb_dim)
        candidates_list = torch.empty_like(Yinit).to(Yinit)
        acq_values = torch.empty(num_restarts).to(Yinit)
        # To maximize the acquisition_function.
        for i, x0 in enumerate(Yinit.cpu().numpy()):
            res = minimize(
                fun=f_np_wrapper,
                args=(f,),
                x0=x0,
                method='trust-constr', #  'SLSQP','trust-constr'
                jac=True,
                constraints=[inequality_constraints],
                options={'maxiter': 100}
            )
            cand_npy = res.x

            # # SLSQP sometimes fails in the line search or may just fail to find a feasible
            # # candidate in which case we just return the starting point. This happens rarely,
            # # so it shouldn't be an issue given enough restarts.
            # nlc = torch.from_numpy(cand_npy).unsqueeze(0).to(B) @ B.T
            # if (nlc > 1).any() or (nlc < -1).any():
            #     cand_npy = x0
            #     logger.warn(
            #         "SLSQP failed to converge to a solution the satisfies the non-linear "
            #         "constraints. Returning the feasible starting point."
            #     )

            candidates = torch.from_numpy(cand_npy).to(dtype=dtype, device=device).view(-1, 1, emb_dim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                with torch.no_grad():
                    acq_val = acq_function(candidates)
            candidates_list[i] = candidates.view(-1)
            acq_values[i] = acq_val

        best = torch.argmax(acq_values)
        batch_candidates = candidates_list[best: best+1]
        batch_acq_values = acq_values[best: best+1]


    return batch_candidates, batch_acq_values


def kdr_bo(eval_func, D, d, n_iterations, n_init=10, batch_size=1):
    wallclocks = torch.zeros(n_init + n_iterations * batch_size)
    startT = time.monotonic()
    """Assume x in [-1, 1]^D"""
    X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    X = 2 * X - 1  # map to [-1, 1]^D
    # logger.info(f"Best initial point: {Y.max().item():.3f}")
    wallclocks[:n_init] = time.monotonic() - startT
    for iter in range(n_iterations):
        train_Y = (Y - Y.mean()) / Y.std()
        B = kdr_sample(X, train_Y, d)
        # if np.linalg.matrix_rank(B) < d:
        #     print("Warning: The projection matrix is not full rank")
        # project down
        x_embedded = X @ B  # shape(n,d_e)

        t2 = time.monotonic()
        # fit model
        gp = SingleTaskGP(train_X=x_embedded, train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, 1e-6).to(train_Y),
            # covar_module=ScaleKernel(RBFKernel())
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        t3 = time.monotonic()
        # Construct an acquisition function
        ei = ExpectedImprovement(model=gp, best_f=train_Y.max())
        # ucb = UpperConfidenceBound(gp, beta=0.1)
        # Optimize the acquisition function
        # linear_constraint = LinearConstraint(B.cpu().numpy(), -1 * np.ones(D), np.ones(D))
        linear_constraint = LinearConstraint(B.cpu().numpy(), -1 * np.ones(D), np.ones(D), keep_feasible=True)

        candidates, _ = acqf_optimizer(
            acq_function=ei,
            n=batch_size,
            inequality_constraints=linear_constraint,
            num_restarts=10,
            raw_samples=1000,
            B=B, projector=B @ B.T,
        )
        t4 = time.monotonic()

        # Project up
        Xopt = candidates @ B.T
        # Sometimes numerical tolerance can have Xopt epsilon outside [0, 1],
        # so clip it back.
        Xopt = torch.clamp(Xopt, min=-1.0, max=1.0)
        Y_next = torch.tensor(
            [eval_func((x+1)/2) for x in Xopt], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        wallclocks[n_init+iter*batch_size] = time.monotonic() - startT
        X = torch.cat((X, Xopt), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        # logger.info(f"Runned trial {iter}. GP: {(t3-t2)/60:.2f}. Max acqf: {(t4-t3)/60:.2f}min")
    return (X+1)/2, Y, wallclocks  # map to [0,1]^D

if __name__ == "__main__":
    import numpy as np

    DIM = 100
    EM_DIM = 4
    from botorch.test_functions import Branin
    branin = Branin().to(dtype=dtype, device=device)

    def branin_emb(x):
        """x is assumed to be in [0, 1]^d"""
        lb, ub = branin.bounds
        return branin(lb + (ub - lb) * x[..., :2]) * -1  # Flip the value for minimization

    X, Y = kdr_bo(branin_emb, D=DIM, d=EM_DIM, n_init=5, n_iterations=20)
    Y_np = -1 * Y
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(Y_np))
    ax.plot([0, len(Y_np)], [0.398, 0.398], "--", c="g", lw=3, label="Optimal value")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found')
    plt.savefig("results.png")