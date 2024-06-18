import torch
from torch.quasirandom import SobolEngine
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sobol(eval_func, ndims, total_trials, device=torch.device("cpu"), dtype=torch.double):
    wallclocks = torch.zeros(total_trials)
    startT = time.monotonic()
    X = SobolEngine(dimension=ndims, scramble=True).draw(total_trials).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    wallclocks[:] = time.monotonic() - startT
    return X, Y, wallclocks