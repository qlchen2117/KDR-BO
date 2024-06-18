import numpy as np
# from botorch.test_functions import Ackley, Griewank, Levy
from pathlib import Path
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

# func, DIM = Hartmann(), 100, 6
# func, DIM, EM_DIM = , 30, 5
# func, DIM, EM_DIM = , 30, 9
# fig_p = Path(f"figures/{type(func).__name__}.png")

def plot_data(data, label):
    print(f"{label}: {data.shape}")
    ax.plot(np.minimum.accumulate(data, axis=1).mean(0), label=label)

def plotLinsub(func_name='Ackley', DIM=60, EM_DIM=10):
    if not func_name in ['Ackley', 'Levy', 'Griewank']:
        raise NotImplementedError

    res_p = Path(f"results/{func_name}-linsub/")

    ## HDBO
    # store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-turbo.npy")), axis=0)
    # ax.plot(np.minimum.accumulate(store_data), label="TURBO")
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-rembo.npy")),
    #     label="REMBO"
    # )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-turbo.npy")),
        label='TuRBO'
    )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-turboD.npy")),
        label='TuRBO-D'
    )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-alebo.npy")),
        label="ALEBO"
    )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-sobol.npy")),
        label="Sobol"
    )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-sir_bo.npy")),
        label="SIR-BO"
    )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-kdr_bo.npy")),
    #     label="KDR-BO"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-mkdr_bo.npy")),
    #     label="MKDR-BO"
    # )
    ax.grid(True)
    ax.set_title(f"{func_name} (D={DIM}, d={EM_DIM})", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    # ax.set_ylim([0, 600])
    ax.legend()
    # plt.savefig(fig_p)
    plt.show()
# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-mkdr_bo2.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="MKDR-BO2")


def plotNlinsub(func_name='Ackley', DIM=60, EM_DIM=10):
    if not func_name in ['Ackley', 'Levy', 'Griewank']:
        raise NotImplementedError

    res_p = Path(f"results/{func_name}-Nlinsub/")

    ## HDBO

    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-rembo.npy")),
    #     label="REMBO"
    # )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-turbo.npy")),
        label='TuRBO'
    )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-HEBO.npy")),
    #     label='HEBO'
    # )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-turboD.npy")),
        label='TuRBO-D'
    )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-alebo.npy")),
    #     label="ALEBO"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-sobol.npy")),
    #     label="Sobol"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-sir_bo.npy")),
    #     label="SIR-BO"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-kdr_bo.npy")),
    #     label="KDR-BO"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-mkdr_bo.npy")),
    #     label="MKDR-BO"
    # )
    # plot_data(
    #     data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-lp.npy")),
    #     label="LP"
    # )
    ax.grid(True)
    ax.set_title(f"{func_name} (D={DIM}, d={EM_DIM})", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    # ax.set_ylim([0, 600])
    ax.legend()
    # plt.savefig(fig_p)
    plt.show()

# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-hebo.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="HEBO")
# store_data = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-manifold_bo.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="MBO")
# ax.plot([0, len(store_data)], [func.optimal_value, func.optimal_value], "k--", lw=3)


if __name__ == '__main__':
    # plotLinsub('Ackley', 100)
    # plotLinsub('Levy')
    # plotLinsub('Griewank')
    # plotNlinsub('Ackley', 100)
    plotNlinsub('Levy', 100)
    # plotNlinsub('Griewank', 100)