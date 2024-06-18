# Define a function to be optimized.
# Here we use a simple synthetic function with a d=2 true linear embedding, in
# a D=100 ambient space.
import numpy as np
import torch
from botorch.exceptions.warnings import OptimizationWarning, InputDataWarning
import warnings
# warnings.filterwarnings("ignore", category=OptimizationWarning)
# warnings.filterwarnings("ignore", category=InputDataWarning)
from ax.modelbridge.strategies.alebo import ALEBOStrategy

def alebo(eval_func, D, d, n_init, total_trials):

    # Define the parameters in the format expected by Ax.
    parameters = [
        {"name": f"x{i}", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
        for i in range(D)
    ]
    # Setup the ALEBO optimization strategy
    alebo_strategy = ALEBOStrategy(D=D, d=d, init_size=n_init)

    # Run the optimization loop with that strategy
    # This will take about 30 mins to run

    from ax.service.managed_loop import optimize

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        objective_name="objective",
        evaluation_function=eval_func,
        total_trials=total_trials,
        generation_strategy=alebo_strategy,
    )
    Y = np.array([trial.objective_mean for trial in experiment.trials.values()])
    X = np.array(
        [np.array([trial.arm.parameters.get(f"x{i}") for i in range(D)])
        for trial in experiment.trials.values()]
    )
    return X, Y

# if __name__ == "__main__":
#     from ax.utils.measurement.synthetic_functions import branin
#     import numpy as np

#     DIM = 100
#     EM_DIM = 4
#     def branin_evaluation_function(parameterization):
#         # Evaluates Branin on the first two parameters of the parameterization.
#         # Other parameters are unused.
#         lb, ub = np.full((DIM,), -5), np.full((DIM,), 10)
#         lb[1], ub[1] = 0, 15
#         x = np.array([parameterization.get(f"x{i}") for i in range(len(parameterization))])
#         x = lb + (ub - lb) * x
#         return {"objective": (-1 * branin(x[..., :2]), 0.0)}

#     X, Y = alebo(branin_evaluation_function, D=DIM, d=EM_DIM, n_init=5, total_trials=30)
#     Y_np = -1 * Y
#     from matplotlib import pyplot as plt

#     fig = plt.figure(figsize=(12, 6))
#     ax = fig.add_subplot(111)
#     ax.grid(alpha=0.2)
#     ax.plot(range(1, 31), np.minimum.accumulate(Y_np))
#     ax.axhline(y=branin.fmin, ls='--', c='k')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Best objective found')
#     plt.savefig("results.png")

if __name__ == "__main__":
    DIM = 20
    EFFECT_DIM = 10
    EM_DIM = EFFECT_DIM
    N_INIT = 5
    TOTAL_TRIALS = 100

    from botorch.test_functions import Ackley
    fun = Ackley(dim=EFFECT_DIM)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)

    def eval_objective(parameterization):
        """This is a helper function we use to unnormalize and evalaute a point"""
        lb, ub = fun.bounds
        x = torch.tensor([parameterization.get(f"x{i}") for i in range(EFFECT_DIM)]).to(lb)
        x = x[:EFFECT_DIM]
        x = lb + (ub - lb) * x
        y = fun(x).item() * -1
        return {"objective": (y, 0.0)}
    X, Y = alebo(eval_objective, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
    Y_np = -1 * Y
    import numpy as np
    import matplotlib.pyplot as plt
    fx = np.minimum.accumulate(Y_np)
    plt.plot(fx, marker="", lw=3)

    plt.plot([0, len(Y_np)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
    plt.show()