# Define a function to be optimized.
# Here we use a simple synthetic function with a d=2 true linear embedding, in
# a D=100 ambient space.
import torch
from botorch.exceptions.warnings import OptimizationWarning, InputDataWarning
import warnings
from botorch.utils import standardize
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)
from .alebo_simple import ALEBO
from .alebo_initializer import ALEBOInitializer
import time

def gen_projection(d: int, D: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Generate the projection matrix B as a (d x D) tensor"""
    B0 = torch.randn(d, D, dtype=dtype, device=device)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    B = B0 / torch.sqrt((B0**2).sum(dim=0))
    return B


def alebo(eval_func, D, d, n_init, total_trials, dtype=torch.double, device=torch.device("cpu")):
    wallclocks = torch.zeros(total_trials)
    B = gen_projection(d=d, D=D, device=device, dtype=dtype)
    initializer = ALEBOInitializer(B=B.cpu().numpy())
    startT = time.monotonic()
    X_bo, _ = initializer.gen(n=n_init, bounds=[(-1.0, 1.0)] * D)
    X_bo = torch.from_numpy(X_bo).to(B)
    Y_bo = torch.tensor(
        [eval_func((x + 1) / 2) for x in X_bo], dtype=dtype, device=device
    ).unsqueeze(-1)
    wallclocks[:n_init] = time.monotonic() - startT
    bounds = [(-1, 1)] * D
    while len(Y_bo) < total_trials:
        alebo = ALEBO(B=B, laplace_nsamp=25, fit_restarts=1)
        alebo.fit([X_bo], [standardize(Y_bo)], [torch.full_like(Y_bo, 1e-6).to(Y_bo)],
                   bounds=bounds)
        x_next = alebo.gen(n=1, bounds=bounds)
        y_next = torch.tensor(
            [eval_func((x + 1) / 2) for x in x_next], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        wallclocks[len(Y_bo)] = time.monotonic() - startT
        X_bo = torch.cat((X_bo, x_next), dim=0)
        Y_bo = torch.cat((Y_bo, y_next), dim=0)
    return X_bo, Y_bo, wallclocks

def test():
    import numpy as np
    DIM = 100
    EM_DIM = 4
    dtype, device = torch.double, torch.device("cpu")
    from botorch.test_functions import Branin
    branin = Branin().to(dtype=dtype, device=device)

    def branin_emb_min(x):
        """x is assumed to be in [0, 1]^d"""
        lb, ub = branin.bounds
        return branin(lb + (ub - lb) * x[..., :2]).item()  # Flip the value for minimization

    _, Y, _ = alebo(branin_emb_min, D=DIM, d=EM_DIM, n_init=5, total_trials=30)
    Y_np = Y.cpu().numpy()
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(Y_np))
    ax.plot([0, len(Y_np)], [0.398, 0.398], "--", c="g", lw=3, label="Optimal value")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found')
    plt.show()
