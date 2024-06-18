import numpy as np
from torch import Tensor
import LassoBench
from pathlib import Path
res_p = Path("results/lassobench/")  # Save Results
if not res_p.exists():
    import os
    os.makedirs(str(res_p))

def testLasso(pick_data='synt_high', N_EPOCH=10, N_ITERACTIONS=300, N_INIT=15, BATCH_SIZE=1):
    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE

    if pick_data == 'synt_high':
        lassoBench = LassoBench.SyntheticBenchmark(pick_bench=pick_data)
    elif pick_data == 'DNA':
        lassoBench = LassoBench.RealBenchmark(pick_data=pick_data)
    else:
        raise NotImplementedError
    DIM = lassoBench.n_features
    EM_DIM = 10

    lb, ub = -1 * np.ones(DIM), np.ones(DIM)

    def eval_objective(x: Tensor):
        x_np = lb + (ub - lb) * x.cpu().numpy()
        loss = lassoBench.evaluate(x_np)
        return loss * -1  # Flip the value for minimization

    def eval_objective4min(x: Tensor):
        x_np = lb + (ub - lb) * x.cpu().numpy()
        return lassoBench.evaluate(x_np)
 
    def eval_objective4alebo(parameterization):  # wrapper for alebo
        """x is assumed to be in [0, 1]^d"""
        x_np = np.array([parameterization.get(f"x{i}") for i in range(DIM)])
        y = lassoBench.evaluate(x_np) * -1 # Flip the value for minimization
        return {"objective": (y, 0.0)}

    def eval_objective4hebo(x: np.ndarray):  # wrapper for hebo
        """x is assumed to be in [0, 1]^d"""
        return lassoBench.evaluate(lb + (ub - lb) * x)

    class Objective:  # for turbo
        def __init__(self, obj) -> None:
            self.obj = obj
        def __call__(self, x: np.ndarray):
            return self.obj.evaluate(x)

    store_data = np.empty((N_EPOCH, TOTAL_TRIALS))

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

    # from HDBO.alebo_wrap import alebo
    # for i in range(N_EPOCH):
    #     _, Y = alebo(eval_objective4alebo, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    #     Y_np = -1 * Y
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-alebo.npy"), store_data)

    # from HDBO.kdr_bo.kdr_bo import kdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = kdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-kdr_bo.npy"), store_data)

    # from HDBO.mkdr_bo.mkdr_bo import mkdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = mkdr_bo(
    #         eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS,
    #         gp_lr=0.01, batchgp_lr=0.1, acq_restart=5, optim_iter=1
    #     )
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-mkdr_bo.npy"), store_data)

    # from HDBO.mkdr_bo.mkdr_bo_scale import mkdr_bo_scale
    # for i in range(N_EPOCH):
    #     _, Y = mkdr_bo_scale(
    #         eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS,
    #         gp_lr=0.01, batchgp_lr=0.1, acq_restart=5, optim_iter=30
    #     )
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-mkdr_bo_scale.npy"), store_data)

    # from turbo import Turbo1
    # func = Objective(lassoBench)
    # for i in range(N_EPOCH):
    #     turbo1 = Turbo1(
    #         f=func,  # Handle to objective function
    #         lb=lb,  # Numpy array specifying lower bounds
    #         ub=ub,  # Numpy array specifying upper bounds
    #         n_init=N_INIT,  # Number of initial bounds from an Latin hypercube design
    #         max_evals=TOTAL_TRIALS,  # Maximum number of evaluations
    #         batch_size=BATCH_SIZE,  # How large batch size TuRBO uses
    #         verbose=False,  # Print information from each batch
    #         use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    #         max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    #         n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    #         min_cuda=1024,  # Run on the CPU for small datasets
    #         device="cpu",  # "cpu" or "cuda"
    #         dtype="float64",  # float64 or float32
    #     )
    #     turbo1.optimize()
    #     store_data[i] = turbo1.fX.ravel()[:TOTAL_TRIALS]  # Observed values
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-turbo.npy"), store_data)

    # from HDBO.turboD_highD import turboD
    # # from HDBO.turboD import turboD
    # for i in range(N_EPOCH):
    #     _, Y = turboD(
    #         eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, batch_size=BATCH_SIZE,
    #         low_dim=20
    #     )
    #     store_data[i] = Y.view(-1).cpu().numpy()[:TOTAL_TRIALS]
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-turboD.npy"), store_data)

    # from BO.sobol import sobol
    # for i in range(N_EPOCH):
    #     _, Y = sobol(eval_objective, ndims=DIM, total_trials=TOTAL_TRIALS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-sobol.npy"), store_data)

    # from HDBO.sir_bo.sir_bo import sir_bo
    # for i in range(N_EPOCH):
    #     _, Y = sir_bo(eval_objective4hebo, TOTAL_TRIALS, EM_DIM, DIM, N_INIT)
    #     store_data[i] = Y.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-sir_bo.npy"), store_data)

    # from HDBO.sir_bo.ksir_bo import ksir_bo
    # for i in range(N_EPOCH):
    #     _, Y = ksir_bo(eval_objective4hebo, TOTAL_TRIALS, EM_DIM, DIM, N_INIT)
    #     store_data[i] = Y.ravel()
    # np.save(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-ksir_bo.npy"), store_data)

    from HDBO.add_bo_quasi_nt.BOLibkky.addGPBO import add_gp_bo
    from HDBO.add_bo_quasi_nt.BOLibkky.preprocessDecomposition import HyperParam
    for i in range(N_EPOCH):
        _, _, _, boVals, _ = add_gp_bo(
            eval_objective,
            bounds=np.array([[0., 1.]]*DIM),
            num_iterations=TOTAL_TRIALS,
            n_init=N_INIT,
            params=HyperParam(DIM, 4, True)
        )
        store_data[i] = boVals * -1
    np.save(res_p.joinpath(f"{pick_data}-D{DIM}-add_bo.npy"), store_data)

## HDBO
# np.save(res_p.joinpath(f"lasso-dna-D{DIM}-d{EM_DIM}-turbo.npy"), Y_turbo)
# np.save(res_p.joinpath(f"lasso-dna-D{DIM}-d{EM_DIM}-mkdr_bo.npy"), Y_mkdr_bo)
