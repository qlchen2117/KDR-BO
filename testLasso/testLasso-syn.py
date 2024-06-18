import numpy as np
from torch import Tensor
import LassoBench

N_EPOCH = 7
N_ITERACTIONS = 300
N_INIT = 15
BATCH_SIZE = 1
TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE

pick_data = 'synt_high'
synt_bench = LassoBench.SyntheticBenchmark(pick_bench=pick_data)
DIM = synt_bench.n_features
EM_DIM = 10

lb, ub = -1 * np.ones(DIM), np.ones(DIM)

def eval_objective(x: Tensor):
    x_np = lb + (ub - lb) * x.cpu().numpy()
    loss = synt_bench.evaluate(x_np)
    return loss * -1  # Flip the value for minimization

def eval_objective4alebo(parameterization):  # wrapper for alebo
    """x is assumed to be in [0, 1]^d"""
    x_np = np.array([parameterization.get(f"x{i}") for i in range(DIM)])
    y = synt_bench.evaluate(x_np) * -1 # Flip the value for minimization
    return {"objective": (y, 0.0)}

def eval_objective4hebo(x: np.ndarray):  # wrapper for hebo
    """x is assumed to be in [0, 1]^d"""
    return synt_bench.evaluate(lb + (ub - lb) * x)

class Objective:
    def __init__(self, obj) -> None:
        self.obj = obj
    def __call__(self, x: np.ndarray):
        return self.obj.evaluate(x)

# # Save Results
from pathlib import Path
res_p = Path("results/lassobench/")

store_data = np.empty((N_EPOCH, TOTAL_TRIALS))


    # from HDBO.turbo import turbo
    # _, Y = turbo(eval_func=eval_objective, dim=DIM, n_iterations=N_ITERACTIONS, n_init=N_INIT, batch_size=BATCH_SIZE)
    # Y_np = -1 * Y.cpu().numpy()
    # Y_turbo[i] = Y_np.squeeze()

    # from HDBO.saasbo import saasbo
    # X, Y = saasbo(eval_func=eval_objective, ndims=DIM, n_iterations=N_ITERACTIONS//5, n_init=N_INIT, batch_size=5)
    # Y_np = -1 * Y.cpu().numpy()
    # ax.plot(np.minimum.accumulate(Y_np), label="SAASBO")

# from HDBO.rembo import rembo
# for i in range(N_EPOCH):
#     _, Y = rembo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
#     Y_np = -1 * Y.cpu().numpy()
#     store_data[i] = Y_np.ravel()
# np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-rembo.npy"), store_data)

from HDBO.alebo_wrap import alebo
for i in range(N_EPOCH):
    _, Y = alebo(eval_objective4alebo, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    Y_np = -1 * Y
    store_data[i] = Y_np.ravel()
np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-alebo.npy"), store_data)

from HDBO.kdr_bo.kdr_bo import kdr_bo
for i in range(N_EPOCH):
    _, Y = kdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    Y_np = -1 * Y.cpu().numpy()
    store_data[i] = Y_np.ravel()
np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-kdr_bo.npy"), store_data)

    # from HDBO.mkdr_bo.mkdr_bo import mkdr_bo
    # _, Y = mkdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    # Y_np = -1 * Y.cpu().numpy()
    # Y_mkdr_bo[i] = Y_np.squeeze()

from turbo import Turbo1
func = Objective(synt_bench)
for i in range(N_EPOCH):
    turbo1 = Turbo1(
        f=func,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=N_INIT,  # Number of initial bounds from an Latin hypercube design
        max_evals=TOTAL_TRIALS,  # Maximum number of evaluations
        batch_size=BATCH_SIZE,  # How large batch size TuRBO uses
        verbose=False,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    turbo1.optimize()
    store_data[i] = turbo1.fX.ravel()[:TOTAL_TRIALS]  # Observed values
np.save(res_p.joinpath(f"{pick_data}-D{DIM}-turbo.npy"), store_data)

## HDBO
# np.save(res_p.joinpath(f"lasso-dna-D{DIM}-d{EM_DIM}-turbo.npy"), Y_turbo)
# np.save(res_p.joinpath(f"lasso-dna-D{DIM}-d{EM_DIM}-mkdr_bo.npy"), Y_mkdr_bo)
