import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


def hebo_optimize(obj, ndims, n_init, total_trials, batch_size=1):
    params = [{'type': 'num', "name": f"x{i}", "lb": 0, "ub": 1}
              for i in range(ndims)]

    space = DesignSpace().parse(params)
    opt = HEBO(space, rand_sample=n_init)
    for _ in range(total_trials):
        rec = opt.suggest(batch_size)
        values = np.empty((batch_size, 1))
        for j in range(batch_size):
            values[j, :] = obj(rec.values[j, :])
        opt.observe(rec, values)
    return opt.X.values, opt.y


if __name__ == '__main__':
    import torch
    from botorch.test_functions import Branin, Ackley
    N_EPOCH = 5
    DIM = 100
    EM_DIM = 2
    N_ITERACTIONS = 150
    N_INIT = 10
    BATCH_SIZE = 1
    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE

    func = Ackley(dim=EM_DIM)
    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)

    def eval_objective4hebo(x):  # wrapper for hebo
        """x is assumed to be in [0, 1]^d"""
        lb, ub = func.bounds
        x = torch.tensor(x[:EM_DIM]).to(lb)
        x = lb + (ub - lb) * x
        y = func(x).item()
        return y

    Y_hebo = np.empty((N_EPOCH, TOTAL_TRIALS))   
    for i in range(N_EPOCH):
        # Call HEBO: to minimize a function
        _, Y = hebo_optimize(eval_objective4hebo, ndims=DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
        Y_hebo[i] = Y.squeeze()

    # Read results
    Y_hebo = np.mean(Y_hebo, axis=0)

    # Draw pictures
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.minimum.accumulate(Y_hebo), label="HEBO")

    ax.plot([0, TOTAL_TRIALS], [func.optimal_value, func.optimal_value], "k--", lw=3)

    ax.grid(True)
    ax.set_title(f"{type(func).__name__}, D = {DIM}", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    # ax.set_ylim([0, 8])
    ax.legend()
    plt.show()
