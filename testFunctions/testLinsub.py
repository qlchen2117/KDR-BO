import json
import numpy as np
import torch
from botorch.test_functions import Ackley, Levy, Griewank
from scipy.stats import special_ortho_group

# import logging
# logging.getLogger().setLevel(logging.INFO)
from pathlib import Path

dir = Path(__file__).parent
def generate_rotation_mat(dim, emb_dim):
    np.random.seed(1000)
    random_basis = special_ortho_group.rvs(dim)[:emb_dim, :]
    with open(dir.joinpath(f'data/random_subspace_{dim}x{emb_dim}.json'), 'w') as fout:
        json.dump(random_basis.tolist(), fout)

def testLinsub(func_name='Ackley', N_EPOCH = 3, DIM = 60, EM_DIM = 10, N_ITERACTIONS = 100, N_INIT = 15, BATCH_SIZE = 1):
    with open(dir.joinpath(f'data/random_subspace_{DIM}x{EM_DIM}.json'), 'r') as fin:
        random_basis = torch.tensor(json.load(fin)).to(dtype=torch.double)

    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE
    if func_name == 'Ackley':
        func = Ackley(dim=EM_DIM)
        func.bounds[0, :].fill_(-5)
        func.bounds[1, :].fill_(10)
    elif func_name == 'Levy':
        func = Levy(dim=EM_DIM)
        func.bounds[0, :].fill_(-5)
        func.bounds[1, :].fill_(10)
    elif func_name == 'Griewank':
        func = Griewank(EM_DIM)
        func.bounds[0, :].fill_(-300.)
        func.bounds[1, :].fill_(600.)
    else:
        raise NotImplementedError

    res_p = Path(f"results/{type(func).__name__}-linsub")
    if not res_p.exists():
        import os
        os.makedirs(str(res_p))
    lb, ub = func.bounds
    def eval_objective(xx):
        """x is assumed to be in [0, 1]^D"""
        zz = random_basis @ xx
        zz = (zz + 1) / 2.  # zz \in [-1, 1], so adjust zz to be closer to [0, 1], and evaluate
        return func(lb + (ub - lb) * zz).item() * -1

    def eval_objective4min(xx):
        """x is assumed to be in [0, 1]^D"""
        zz = random_basis @ xx
        zz = (zz + 1) / 2.  # zz \in [-1, 1], so adjust zz to be closer to [0, 1], and evaluate
        return func(lb + (ub - lb) * zz).item()

    def eval_objective_wrap(parameterization):  # wrapper for alebo
        """x is assumed to be in [0, 1]^D"""
        xx = torch.tensor([parameterization.get(f"x{i}") for i in range(DIM)]).to(lb)

        zz = random_basis @ xx
        zz = (zz + 1) / 2.  # zz \in [-1, 1], so adjust zz to be closer to [0, 1], and evaluate
        y = func(lb + (ub - lb) * zz).item() * -1 # Flip the value for minimization
        return {"objective": (y, 0.0)}

    def eval_objective4hebo(x):  # wrapper for hebo
        """x is assumed to be in [0, 1]^D"""
        xx = torch.from_numpy(x).to(lb)

        zz = random_basis @ xx
        zz = (zz + 1) / 2.  # zz \in [-1, 1], so adjust zz to be closer to [0, 1], and evaluate
        return func(lb + (ub - lb) * zz).item()

    store_data = np.empty((N_EPOCH, TOTAL_TRIALS))

    # from HDBO.saasbo import saasbo
    # X, Y = saasbo(eval_func=eval_objective, ndims=DIM, n_iterations=N_ITERACTIONS//5, n_init=N_INIT, batch_size=5)
    # Y_np = -1 * Y.cpu().numpy()

    # from HDBO.rembo import rembo
    # for i in range(N_EPOCH):
    #     _, Y = rembo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-rembo.npy"), store_data)

    # from HDBO.alebo_wrap_old import alebo
    # for i in range(N_EPOCH):
    #     _, Y = alebo(eval_objective_wrap, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    #     Y_np = -1 * Y
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-alebo.npy"), store_data)

    # from BO.hebo_wrapper import hebo_optimize
    # for i in range(N_EPOCH):
    #     _, Y_np = hebo_optimize(eval_objective4hebo, ndims=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM-d{EM_DIM}}-hebo.npy"), store_data)

    # from BO.bo import bo
    # for i in range(N_EPOCH):
    #     _, Y = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, use_input_warp=True, opt_acq='MACE')
    #     Y_np = Y.cpu().numpy() * -1
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo_moo.npy"), store_data)

    # from HDBO.kdr_bo.kdr_bo import kdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = kdr_bo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-kdr_bo.npy"), store_data)

    # from HDBO.mkdr_bo.mkdr_bo import mkdr_bo
    # for i in range(N_EPOCH):
    #     _, Y = mkdr_bo(
    #         eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS,
    #         batchgp_lr=0.01
    #     )
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-mkdr_bo.npy"), store_data)

    # from turbo import Turbo1
    # for i in range(N_EPOCH):
    #     turbo1 = Turbo1(
    #         f=eval_objective4hebo,  # Handle to objective function
    #         lb=np.zeros(DIM),  # Numpy array specifying lower bounds
    #         ub=np.ones(DIM),  # Numpy array specifying upper bounds
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
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-turbo.npy"), store_data)

    # from HDBO.turboD_highD import turboD
    # for i in range(N_EPOCH):
    #     _, Y = turboD(eval_func=eval_objective4min, dim=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS, batch_size=BATCH_SIZE,
    #                   low_dim=20)
    #     store_data[i] = Y.view(-1).cpu().numpy()[:TOTAL_TRIALS]
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-turboD.npy"), store_data)

    # from BO.sobol import sobol
    # for i in range(N_EPOCH):
    #     _, Y = sobol(eval_objective, ndims=DIM, total_trials=TOTAL_TRIALS)
    #     Y_np = -1 * Y.cpu().numpy()
    #     store_data[i] = Y_np.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-sobol.npy"), store_data)

    # from HDBO.sir_bo.sir_bo import sir_bo
    # for i in range(N_EPOCH):
    #     _, Y = sir_bo(eval_objective4hebo, TOTAL_TRIALS, EM_DIM, DIM, N_INIT)
    #     store_data[i] = Y.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-sir_bo.npy"), store_data)

    # from HDBO.sir_bo.ksir_bo import ksir_bo
    # for i in range(N_EPOCH):
    #     _, Y = ksir_bo(eval_objective4hebo, TOTAL_TRIALS, EM_DIM, DIM, N_INIT)
    #     store_data[i] = Y.ravel()
    # np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-ksir_bo.npy"), store_data)

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
    np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-add_bo.npy"), store_data)

# from HDBO.mybo import manifold_bo
# for i in range(N_EPOCH):
#     _, Y = manifold_bo(eval_objective, dim=DIM, emb_dim=EM_DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS)
#     Y_np = -1 * Y.cpu().numpy()
#     store_data[i] = Y_np.ravel()
# np.save(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-manifold_bo.npy"), store_data)
