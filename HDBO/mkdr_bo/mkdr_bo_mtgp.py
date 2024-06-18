import logging
import torch
from torch import Tensor
from typing import Callable, Tuple
from torch.quasirandom import SobolEngine
import time
from functools import partial
from torch.optim.adam import Adam
from botorch.utils.types import DEFAULT
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.exceptions.warnings import InputDataWarning

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import warnings
# warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
# from scipy.optimize._hessian_update_strategy import SR1
# from datafold import LaplacianPyramidsInterpolator
from .mKDR import MKDR
from .mkdr_bo_initializer import gen

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
logger.addHandler(streamHandler)

device = torch.device("cpu")
dtype = torch.double
ETA = 0.5

def acqf_optimizer(
    Xrnd_npy,
    mkdr, batchgp,
    acq_function,
    high_dim: int, emb_dim: int,
    num_restarts: int, optim_iter: int
):
    # noise = torch.randn(1, dtype=dtype, device=device)
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

    def cons_f(embed_test: np.ndarray):
        embed_test = torch.from_numpy(embed_test[np.newaxis, :]).to(dtype=dtype, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn = batchgp.posterior(embed_test.unsqueeze(1))  # batch_size x q x dim --> batch_size x q x n_output
            mean, stddev = mvn.mean.view(-1), mvn.stddev.view(-1)
            return torch.cat((mean - stddev, mean + stddev), dim=0).cpu().numpy()

    def cons_J(embed_test: np.ndarray):
        xx = torch.from_numpy(embed_test).to(dtype=dtype, device=device).contiguous().requires_grad_(True)
        def up_projection(zz):
            mvn = batchgp.posterior(zz.unsqueeze(0))
            mean, stddev = mvn.mean.view(-1), mvn.stddev.view(-1)
            return torch.cat((mean - stddev, mean + stddev), dim=0)
        return torch.autograd.functional.jacobian(up_projection, xx).cpu().numpy()

    Zinit = Zinit.squeeze(1)  # (b, emb_dim)
    candidates_list = torch.empty_like(Zinit).to(Zinit)
    acq_values = torch.empty(num_restarts).to(Zinit)
    # nonlinear_constraint = NonlinearConstraint(
    #     cons_f, np.hstack((-np.ones(Zinit.shape[-1]), 0)), np.hstack((np.ones(Zinit.shape[-1]), 0.5)),
    #     jac=cons_J, keep_feasible=True)
    nonlinear_constraint = NonlinearConstraint(
        cons_f,
        lb=-1, ub=1,
        # lb=np.hstack((-1 * np.ones(emb_dim), np.full(emb_dim, -np.inf))),
        # ub=np.hstack((np.ones(emb_dim), np.full(emb_dim, np.inf))),
        # jac=cons_J,
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
            options={'maxiter': optim_iter}
        )
        cand_npy = res.x

        # SLSQP sometimes fails in the line search or may just fail to find a feasible
        # candidate in which case we just return the starting point. This happens rarely,
        # so it shouldn't be an issue given enough restarts.
        nlc = cons_f(cand_npy)
        if (nlc > 1).any() or (nlc < -1).any():
            cand_npy = x0
            logging.warn(
                "Trust-constr failed to converge to a solution the satisfies the non-linear "
                "constraints. Returning the feasible starting point."
            )

        candidates = torch.from_numpy(cand_npy).to(dtype=dtype, device=device).view(-1, 1, emb_dim)
        with torch.no_grad():
            acq_val = acq_function(candidates)
        candidates_list[i] = candidates.view(-1)
        acq_values[i] = acq_val

    best = torch.argmax(acq_values)
    batch_candidates = candidates_list[best]
    batch_acq_values = acq_values[best]

    return batch_candidates, batch_acq_values


def mkdr_bo_mtgp(
    eval_func, D, d, n_iterations, n_init=10, batch_size=1,
    gp_lr=0.01, batchgp_lr=0.1, acq_restart=10, optim_iter=30

):
    """Assume x in [-1, 1]^D"""
    X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    X = 2 * X - 1  # map to [-1, 1]^D
    logger.info(f"Best initial point: {Y.max().item():.3f}")
    mkdr = MKDR()
    optimizer_gp = partial(Adam, lr=gp_lr)
    optimizer_batchgp = partial(Adam, lr=batchgp_lr)

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
        # fit_gpytorch_mll(mll)
        fit_gpytorch_mll_torch(
            mll,
            optimizer=optimizer_gp,
            # step_limit=3000,
            stopping_criterion=DEFAULT
        )
        # Construct an acquisition function
        ei = qExpectedImprovement(model=gp, best_f=train_Y.max())

        t2 = time.monotonic()

        # use MTGP to project up
        batchgp = KroneckerMultiTaskGP(x_embedded, X)
        batch_mll = ExactMarginalLogLikelihood(batchgp.likelihood, batchgp)
        fit_gpytorch_mll_torch(
            batch_mll,
            optimizer=optimizer_batchgp,
            # step_limit=3000,
            stopping_criterion=DEFAULT
        )

        t3 = time.monotonic()
        # Optimize the acquisition function
        num_gen, Xrnd_npy = gen(mkdr, batchgp, n=1024, dim=D, dtype=dtype, device=device)
        candidate, _ = acqf_optimizer(
            Xrnd_npy,
            mkdr, batchgp, ei,
            high_dim=D, emb_dim=d,
            num_restarts=acq_restart, optim_iter=optim_iter
        )
        t4 = time.monotonic()

        # Project up
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Xopt = batchgp.posterior(candidate.view(-1, d)).mean

        # Sometimes numerical tolerance can have Xopt epsilon outside [-1, 1],
        # so clip it back.
        Xopt = torch.clamp(Xopt, min=-1.0, max=1.0)
        Y_next = torch.tensor(
            [eval_func((x+1)/2) for x in Xopt], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X = torch.cat((X, Xopt), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        logger.info(f"Runned trial {iter}. num_gen: {num_gen}. batchGP: {(t3-t2)/60:.2f}. Max acqf: {(t4-t3)/60:.2f}min")
    return (X+1)/2, Y  # map to [0,1]^D
