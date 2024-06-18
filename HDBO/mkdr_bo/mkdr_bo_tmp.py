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
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize, Bounds
from scipy.optimize._hessian_update_strategy import SR1
# from datafold import LaplacianPyramidsInterpolator
from .mKDR import MKDR
from .mkdr_bo_initializer import gen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double



def acqf_optimizer(
    mkdr, batchgp,
    acq_function,
    high_dim: int, emb_dim: int,
    num_restarts: int,
):
    noise = torch.randn(1, dtype=dtype, device=device)
    Xrnd_npy = gen(mkdr, batchgp, n=1000, dim=high_dim, noise=noise, dtype=dtype, device=device)
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
        batchgp.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn = batchgp(embed_test)
            mean, stddev = mvn.mean.view(-1), mvn.variance.view(-1).sqrt()
            return (mean + stddev * noise).cpu().numpy()

    def cons_J(embed_test: np.ndarray):
        xx = torch.from_numpy(embed_test).to(dtype=dtype, device=device).contiguous().requires_grad_(True)
        def up_projection(zz):
            batchgp.eval()
            mvn = batchgp(zz.unsqueeze(0))
            mean, stddev = mvn.mean.view(-1), mvn.variance.view(-1).sqrt()
            return mean + stddev * noise
        return torch.autograd.functional.jacobian(up_projection, xx).cpu().numpy()

    Zinit = Zinit.squeeze(1)  # (b, emb_dim)
    candidates_list = torch.empty_like(Zinit).to(Zinit)
    acq_values = torch.empty(num_restarts).to(Zinit)
    nonlinear_constraint = NonlinearConstraint(cons_f, -1, 1, jac=cons_J, hess=SR1(), keep_feasible=True)
    # To maximize the acquisition_function.
    for i, x0 in enumerate(Zinit.cpu().numpy()):
        res = minimize(
            fun=f_np_wrapper,
            args=(f,),
            x0=x0,
            method='trust-constr', #  'SLSQP','trust-constr'
            jac=True,
            constraints=[nonlinear_constraint],
            options={'maxiter': 100}
        )
        cand_npy = res.x

        # SLSQP sometimes fails in the line search or may just fail to find a feasible
        # candidate in which case we just return the starting point. This happens rarely,
        # so it shouldn't be an issue given enough restarts.
        nlc = cons_f(cand_npy)
        if (nlc > 1).any() or (nlc < -1).any():
            cand_npy = x0
            logging.warn(
                "SLSQP failed to converge to a solution the satisfies the non-linear "
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


def mkdr_bo(eval_func, D, d, n_iterations, n_init=10, batch_size=1):
    """Assume x in [-1, 1]^D"""
    X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    X = 2 * X - 1  # map to [-1, 1]^D
    print(f"Best initial point: {Y.max().item():.3f}")
    mkdr = MKDR()
    optimizer = partial(Adam, lr=0.01)
    for _ in range(n_iterations):
        train_Y = (Y - Y.mean()) / Y.std()
        # project down
        n_eigenpairs = X.shape[0]
        d = min(d, n_eigenpairs - 1)
        x_embedded = mkdr.fit_transform(X.numpy(), train_Y.numpy(), n_eigenpairs, d)
        x_embedded = torch.from_numpy(x_embedded).to(X)
        # fit model

        gp = SingleTaskGP(train_X=x_embedded, train_Y=train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # Construct an acquisition function
        ei = qExpectedImprovement(model=gp, best_f=train_Y.max())


        # # use MTGP to project up
        # mtgp = KroneckerMultiTaskGP(x_embedded, X)
        # mtgp_mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
        # fit_gpytorch_mll_torch(
        #     mtgp_mll,
        #     # optimizer=optimizer,
        #     stopping_criterion=DEFAULT        
        # )
        batchgp = SingleTaskGP(x_embedded, X)
        batch_mll = ExactMarginalLogLikelihood(batchgp.likelihood, batchgp)
        fit_gpytorch_mll_torch(
            batch_mll,
            # optimizer=optimizer,
            # step_limit=3000,
            stopping_criterion=DEFAULT
        )


        # use Laplace Pyramid to project up
        # lpi = LaplacianPyramidsInterpolator(residual_tol=1e-1)
        # lpi.fit(x_embedded.numpy(), X.numpy())

        # def cons_f(embed_test: np.ndarray):
        #     return lpi.predict(embed_test[np.newaxis, :]).ravel()


        # Optimize the acquisition function
        candidate, _ = acqf_optimizer(
            mkdr, batchgp, ei,
            high_dim=D, emb_dim=d, num_restarts=10
        )
        # Project up
        batchgp.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Xopt = batchgp(candidate.view(-1, d)).sample()
            Xopt = Xopt.T
        
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
    return (X+1)/2, Y  # map to [0,1]^D


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
