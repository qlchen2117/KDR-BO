import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from .KDR_sample import kdr_sample
from botorch.exceptions.warnings import OptimizationWarning, InputDataWarning

import warnings
# warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def gen(Q, n, init_bound: int = 16):

    # 在[0,1]^D上uniform采样10000个点
    X01 = SobolEngine(dimension=Q.shape[1], scramble=True).draw(10*n).to(dtype=dtype, device=device)

    finished = False
    b = float(init_bound)
    while not finished:
        # Map to [-b, b]
        X_b = 2 * b * X01 - b  # 为什么要扩展到[-b,b]?
        # Project down to B and back up
        X = X_b @ Q.T  # x' = B^+ B x
        # Filter out to points in [-1, 1]^D
        X = X[(X >= -1.0).all(axis=1) & (X <= 1.0).all(axis=1)]
        if X.shape[0] >= n:
            finished = True
        else:
            b = b / 2.0  # Constrict the space
    X = X[:n, :]
    return X

def alebo_acqf_optimizer(
    acq_function,
    n: int,
    inequality_constraints,
    raw_samples: int,
    num_restarts: int,
    B: Tensor,
    Q: Tensor
):
    """
    Optimize the acquisition function for ALEBO.

    We are optimizing over a polytope within the subspace, and so begin each
    random restart of the acquisition function optimization with points that
    lie within that polytope.
    """
    candidate_list, acq_value_list = [], []
    candidates = torch.tensor([], device=B.device, dtype=B.dtype)

    assert n == 1
    for i in range(n):
        # 在[-1,1]^D上均匀采样1000个点
        Xrnd = gen(Q, n=raw_samples)
        Xrnd = Xrnd.unsqueeze(1)
        # 把随机点嵌入低维空间 shape(b, 1, ndims)
        Yrnd = torch.matmul(Xrnd, B.t())  # Project down to the embedding
        
        import gpytorch
        with gpytorch.settings.max_cholesky_size(2000):
            with torch.no_grad():
                alpha = acq_function(Yrnd)  # shape(b,)
            # 在b个随机点中选num_restarts个点，且y越大越可能被选中
            from botorch.optim.initializers import initialize_q_batch_nonneg
            Yinit = initialize_q_batch_nonneg(X=Yrnd, Y=alpha, n=num_restarts)
            inf_bounds = (  # all constraints are encoded via inequality_constraints
                torch.tensor([[-float("inf")], [float("inf")]])
                .expand(2, Yrnd.shape[-1])
                .to(Yrnd)
            )
            inf_bounds = (  # all constraints are encoded via inequality_constraints
                torch.tensor([[-float("inf")], [float("inf")]])
                .expand(2, Yrnd.shape[-1])
                .to(Yrnd)
            )
            # Optimize the acquisition function, separately for each random restart.
            candidate, acq_value = optimize_acqf(  # 最大化采集函数
                acq_function=acq_function,
                bounds=inf_bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=0,
                options={"method": "SLSQP", "batch_limit": 1},
                inequality_constraints=inequality_constraints,
                batch_initial_conditions=Yinit,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
    candidates = torch.cat(candidate_list, dim=-2)

    return candidates, torch.stack(acq_value_list)


def kdr_bo(eval_func, D, d, n_iterations, n_init=10, batch_size=1):
    """Assume x in [-1, 1]^D"""
    X = SobolEngine(dimension=D, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    X = 2 * X - 1  # map to [-1, 1]^D
    print(f"Best initial point: {Y.max().item():.3f}")

    for _ in range(n_iterations):
        train_Y = (Y - Y.mean()) / Y.std()
        B = kdr_sample(X, train_Y, d)
        # if np.linalg.matrix_rank(B) < d:
        #     print("Warning: The projection matrix is not full rank")

        # project down
        x_embedded = X @ B  # shape(n,d_e)
        # fit model
        gp = SingleTaskGP(train_X=x_embedded, train_Y=train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # Construct an acquisition function
        ei = qExpectedImprovement(model=gp, best_f=train_Y.max())
        # ucb = UpperConfidenceBound(gp, beta=0.1)
        # Optimize the acquisition function
        from ax.models.torch.utils import _to_inequality_constraints
        B_T_inv = torch.pinverse(B.T)
        A = torch.cat((B_T_inv, -B_T_inv))
        bb = torch.ones(2 * D, 1, dtype=dtype, device=device)
        linear_constraints = (A, bb)
        embedding_bounds = torch.tensor([[-1e8, 1e8]] * D, dtype=dtype, device=device)
        embedding_bounds = embedding_bounds.transpose(0, 1)
        candidates, acq_value = alebo_acqf_optimizer(
            acq_function=ei,
            n=batch_size,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=linear_constraints
            ),
            num_restarts=10,
            raw_samples=1000,
            B=B.T, Q=B_T_inv @ B.T,
        )
        # Project up
        Xopt = (B_T_inv @ candidates.t()).t()
        # Sometimes numerical tolerance can have Xopt epsilon outside [0, 1],
        # so clip it back.
        Xopt = torch.clamp(Xopt, min=-1.0, max=1.0)
        Y_next = torch.tensor(
            [eval_func((x+1)/2) for x in Xopt], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X = torch.cat((X, Xopt), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
    return (X+1)/2, Y  # map to [0,1]^D

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