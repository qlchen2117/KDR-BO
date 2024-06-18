import logging
import torch
from torch import Tensor
from typing import Callable, Tuple
from torch.quasirandom import SobolEngine

from functools import partial
from torch.optim.adam import Adam
from botorch.utils.types import DEFAULT
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.exceptions.warnings import InputDataWarning

import time
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize, Bounds
# from scipy.optimize._hessian_update_strategy import SR1
from scipy.stats import qmc
from datafold import LaplacianPyramidsInterpolator
from .mKDR import MKDR

import warnings
# warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
logger.addHandler(streamHandler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def gen(lpi, model, n, dim):
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
        Zrand = model.transform(X_b)
        mean = lpi.predict(Zrand)
        # sigma = np.abs(
        #     lpi.predict(model.transform(mean)) - mean
        # )
        sigma = np.linalg.norm(
            lpi.predict(model.transform(mean)) - mean,
            axis=-1, keepdims=True
        )
        X_b = X_b[(mean - 1.96 * sigma > -1.0).all(axis=1) & (mean + 1.96 * sigma < 1.0).all(axis=1)]
        Xrand = np.vstack((Xrand, X_b))
        iter += 1
        if Xrand.shape[0] >= n:
            return iter, Xrand
    print(Xrand.shape)
    raise NotImplementedError


def acqf_optimizer(
    Xrnd_npy, cons_f: Callable,
    mkdr, lpi,
    acq_function,
    high_dim: int, emb_dim: int,
    num_restarts: int, 
    # mse,
):
    Zrand = torch.from_numpy(mkdr.transform(Xrnd_npy)).to(dtype=dtype, device=device).unsqueeze(1)
    with torch.no_grad():
        alpha = acq_function(Zrand)
    Zinit = initialize_q_batch_nonneg(X=Zrand, Y=alpha, n=num_restarts)

    def f_np_wrapper(x: np.ndarray, f: Callable):
        """Given a torch callable, compute value + grad given a numpy array."""
        if np.isnan(x).any():
            raise RuntimeError(
                f"{np.isnan(x).sum()} elements of the {x.size} element array "
                f"`x` are NaN."
            )
        X = (
            torch.from_numpy(x)
            .to(Zinit)
            .view(-1, 1, emb_dim)  # shape(num_restarts x q, emb_dim)
            .contiguous()
            .requires_grad_(True)
        )
        loss = f(X).sum()
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = torch.autograd.grad(loss, X)[0].contiguous().view(-1).cpu().numpy()
        if np.isnan(gradf).any():
            msg = (
                f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                "gradient array `gradf` are NaN. "
                "This often indicates numerical issues."
            )
            if Zrand.dtype != torch.double:
                msg += " Consider using `dtype=torch.double`."
            raise RuntimeError(msg)
        fval = loss.item()
        return fval, gradf

    def f(x):
        return -acq_function(x)

    Zinit = Zinit.squeeze(1)  # (b, emb_dim)
    candidates_list = torch.empty_like(Zinit).to(Zinit)
    acq_values = torch.empty(num_restarts).to(Zinit)
    nonlinear_constraint = NonlinearConstraint(
        cons_f, -1, 1,
        # lb=np.array([-1] * high_dim + [0]),
        # ub=np.array([1] * high_dim + [mse]),
        keep_feasible=True
    )
    # To maximize the acquisition_function.
    for i, x0 in enumerate(Zinit.cpu().numpy()):
        res = minimize(
            fun=f_np_wrapper,
            args=(f,),
            x0=x0,
            method='trust-constr', #  'SLSQP','trust-constr'
            jac=True,
            constraints=[nonlinear_constraint],
            options={'maxiter': 30}
        )
        cand_npy = res.x

        candidates = torch.from_numpy(cand_npy).to(dtype=dtype, device=device).view(-1, 1, emb_dim)
        with torch.no_grad():
            acq_val = acq_function(candidates)
        candidates_list[i] = candidates.view(-1)
        acq_values[i] = acq_val

    best = torch.argmax(acq_values)
    batch_candidates = candidates_list[best]
    batch_acq_values = acq_values[best]

    return batch_candidates, batch_acq_values


def mkdr_bo_lp(eval_func, D, d, n_iterations, n_init=10, batch_size=1):
    """Assume x in [-1, 1]^D"""
    X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    X = 2 * X - 1  # map to [-1, 1]^D
    logger.info(f"Best initial point: {Y.max().item():.3f}")
    mkdr = MKDR()
    optimizer = partial(Adam, lr=0.01)
    for iter in range(n_iterations):
        train_Y = (Y - Y.mean()) / Y.std()
        # project down
        n_eigenpairs = X.shape[0] - 1
        d = min(d, n_eigenpairs - 1)
        x_embedded = mkdr.fit_transform(X.numpy(), train_Y.numpy(), n_eigenpairs, d)
        x_embedded = torch.from_numpy(x_embedded).to(X)
        # fit model

        gp = SingleTaskGP(train_X=x_embedded, train_Y=train_Y, train_Yvar=torch.full_like(train_Y, 1e-6).to(train_Y))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll_torch(
            mll,
            optimizer=optimizer,
            # step_limit=3000,
            stopping_criterion=DEFAULT
        )
        # Construct an acquisition function
        ei = qExpectedImprovement(model=gp, best_f=train_Y.max())

        # use Laplace Pyramid to project up
        # lpi = LaplacianPyramidsInterpolator(residual_tol=1e-2)
        lpi = LaplacianPyramidsInterpolator(auto_adaptive=True)
        x_emb_np, X_np = x_embedded.numpy(), X.numpy()
        lpi.fit(x_emb_np, X_np)
        # mse = ((mkdr.transform(lpi.predict(x_emb_np)) - x_emb_np) ** 2).sum(axis=-1).max()
        def cons_f(embed_test: np.ndarray):
            Zrnd = embed_test.reshape((-1, d))
            mean = lpi.predict(Zrnd)
            # sigma = np.abs(
            #     lpi.predict(mkdr.transform(mean)) - mean
            # )
            sigma = np.linalg.norm(
                lpi.predict(mkdr.transform(mean)) - mean,
                axis=-1, keepdims=True
            )
            return np.hstack(((mean - 1.96 * sigma).ravel(), (mean + 1.96 * sigma).ravel()))

        t3 = time.monotonic()
        num_gen, Xrnd_npy = gen(lpi, mkdr, n=1024, dim=D)
        # Optimize the acquisition function
        candidate, _ = acqf_optimizer(
            Xrnd_npy, cons_f,
            mkdr, lpi, ei,
            high_dim=D, emb_dim=d, num_restarts=10,
            # mse=mse
        )
        t4 = time.monotonic()

        # Project up
        Xopt = torch.from_numpy(lpi.predict(candidate.view(-1, d))).to(dtype=dtype, device=device)

        # Xopt = torch.tensor(
        #     lpi.predict(candidate.view(-1, d).numpy()),
        #     dtype=dtype, device=device
        # )

        # Sometimes numerical tolerance can have Xopt epsilon outside [-1, 1],
        # so clip it back.
        Xopt = torch.clamp(Xopt, min=-1.0, max=1.0)
        Y_next = torch.tensor(
            [eval_func((x+1)/2) for x in Xopt], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X = torch.cat((X, Xopt), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        logger.info(f"Runned trial {iter}. num_gen: {num_gen}. Max acqf: {(t4-t3)/60:.2f}min")
    return (X + 1) / 2, Y  # map to [0,1]^D


# def acqf_optimizer(
#     acq_function,
#     nonlinear_constraints,
#     ndims: int,
#     Yrnd: Tensor,
#     num_restarts: int,
# ):
#     alpha = acq_function(Yrnd)
#     Yinit = initialize_q_batch_nonneg(X=Yrnd, Y=alpha, n=num_restarts)

#     def f_np_wrapper(x: np.ndarray):
#         """Given a torch callable, compute value + grad given a numpy array."""

#         X = torch.from_numpy(x).to(Yinit).view(-1, 1, ndims)
#         loss = acq_function(X).sum() * -1

#         fval = loss.item()
#         return fval

#     Yinit = Yinit.squeeze(1)  # (b, ndims)
#     candidates_list = torch.empty_like(Yinit).to(Yinit)
#     acq_values = torch.empty(Yinit.shape[0])
#     cons_f, lb, ub = nonlinear_constraints
#     def constraints(xx):
#         assert np.all(np.isfinite(xx))
#         yy = cons_f(xx)
#         return np.hstack((lb - yy, yy - ub))
#     cfun = cma.ConstrainedFitnessAL(f_np_wrapper, constraints) 
#     # To maximize the acquisition_function.
#     for i, x0 in enumerate(Yinit.numpy()):
#         _, es = cma.fmin2(
#             cfun, x0, 1,
#             options={'verbose': -9}
#         )
#         # x = cfun.find_feasible(es)
#         candidates = es.result.xfavorite  # the original x-value may be meaningless

#         candidates = torch.tensor(candidates, dtype=dtype, device=device).view(-1, 1, ndims)
#         with torch.no_grad():
#             acq_val = acq_function(candidates)
#         candidates_list[i] = candidates.view(-1)
#         acq_values[i] = acq_val

#     best = torch.argmax(acq_values.view(-1), dim=0)
#     batch_candidates = candidates_list[best]
#     batch_acq_values = acq_values[best]

#     return batch_candidates, batch_acq_values



# def mkdr_bo(eval_func, D, d, n_iterations, n_init=10, batch_size=1):
#     """Assume x in [-1, 1]^D"""
#     lb, ub = np.ones(D) * -1, np.ones(D)
#     X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
#     Y = torch.tensor(
#         [eval_func(x) for x in X], dtype=dtype, device=device
#     ).unsqueeze(-1)
#     X = 2 * X - 1  # map to [-1, 1]^D
#     print(f"Best initial point: {Y.max().item():.3f}")
#     mkdr = MKDR()
#     for _ in range(n_iterations):
#         train_Y = (Y - Y.mean()) / Y.std()
#         # project down
#         n_eigenpairs = min(D, X.shape[0]-2)
#         d = min(d, n_eigenpairs)
#         x_embedded = mkdr.fit_transform(X.numpy(), train_Y.numpy(), n_eigenpairs=n_eigenpairs, d=d)
#         x_embedded = torch.tensor(x_embedded, dtype=dtype, device=device)
#         # fit model
#         gp = SingleTaskGP(train_X=x_embedded, train_Y=train_Y)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         fit_gpytorch_mll(mll)
#         # Construct an acquisition function
#         ei = qExpectedImprovement(model=gp, best_f=train_Y.max())
#         # ucb = UpperConfidenceBound(gp, beta=0.1)

#         # Optimize the acquisition function
#         def fn_wrapper(xx):
#             zz = mkdr.transform(xx[np.newaxis, :])
#             zz = torch.from_numpy(zz).view(1, 1, -1)
#             return ei(zz).item() * -1  # flip the value for maximization
        
#         # Xopt = torch.empty(batch_size, D, dtype=dtype, device=device)
#         # import cma
#         # for i in range(batch_size):
#         #     mean = np.random.uniform(lb, ub)
#         #     x_cmaes, _ = cma.fmin2(
#         #         fn_wrapper, mean, 1,
#         #         options={'bounds':[lb, ub], 'verbose': -9}
#         #     )
#         #     Xopt[i] = torch.from_numpy(x_cmaes)
#         from scipy.optimize import differential_evolution, Bounds
#         res = differential_evolution(fn_wrapper, Bounds(lb, ub))
#         Xopt = torch.from_numpy(res.x[np.newaxis, :])
#         # Sometimes numerical tolerance can have Xopt epsilon outside [-1, 1],
#         # so clip it back.
        
#         Y_next = torch.tensor(
#             [eval_func((x+1)/2) for x in Xopt], dtype=dtype, device=device
#         ).unsqueeze(-1)

#         # Append data
#         X = torch.cat((X, Xopt), dim=0)
#         Y = torch.cat((Y, Y_next), dim=0)
#     return (X+1)/2, Y  # map to [0,1]^D
