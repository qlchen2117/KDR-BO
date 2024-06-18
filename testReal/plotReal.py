import numpy as np
from pathlib import Path
# Draw pictures
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

def plotReal(bench_name='nas', N_ITERACTIONS=100, EM_DIM=10):
    def plot_data(data, label, maximize=False):
        print(f"{label}: {data.shape}")
        data = np.mean(data[:, :N_ITERACTIONS], axis=0)
        if maximize:
            ax.plot(np.maximum.accumulate(data * -1), label=label)
        else:
            ax.plot(np.minimum.accumulate(data), label=label)

    if bench_name == 'nas':
        DIM=36
        is_maximize = True
        res_p = Path(f"results/{bench_name}/final")
        ax.set_ylim([0.91, 0.94])
    elif bench_name == 'DNA':
        DIM = 180
        is_maximize = False
        res_p = Path(f"results/lassobench/final")
    else:
        raise NotImplementedError
    # Read results
    ## HDBO
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-turbo.npy")),
        "TuRBO", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-rembo.npy")),
        "REMBO", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-alebo.npy")),
        "ALEBO", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-sir_bo.npy")),
        "SIR-BO", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-sobol.npy")),
        "Sobol", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-kdr_bo.npy")),
        "KDR-BO", is_maximize
    )
    plot_data(
        np.load(res_p.joinpath(f"{bench_name}-D{DIM}-d{EM_DIM}-mkdr_bo.npy")),
        "MKDR-BO", is_maximize
    )

    ax.grid(True)
    ax.set_title(f"{bench_name} (D = {DIM})", fontsize=18)
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_ylabel("Test accuracy", fontsize=18)
    # ax.set_xlim([0, 100])
    ax.legend()
    plt.show()
    # plt.savefig(f"{bench_name}.pdf")

## BO
# Y_hebo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-hebo.npy")), axis=0)
# Y_bo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo.npy")), axis=0)
# Y_bo_warp = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo_warp.npy")), axis=0)
# Y_bo_moo = np.mean(np.load(res_p.joinpath(f"Top2-D{DIM}-moo.npy")), axis=0)
# Y_sobol = np.mean(np.load(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-sobol.npy"), axis=0)


# HDBO
# ax.plot(np.maximum.accumulate(Y_sir_bo), label="SIR-BO")

# BO
# ax.plot(np.maximum.accumulate(Y_hebo), label="HEBO")
# ax.plot(np.maximum.accumulate(Y_bo), label="BO")
# ax.plot(np.maximum.accumulate(Y_bo_warp), label="BO-Warp")
# ax.plot(np.maximum.accumulate(Y_bo_moo), label="BO-MOO")
# ax.plot(np.maximum.accumulate(Y_sobol), label="SOBOL")


# ax.plot([0, len(store_data)], [1, 1], "k--", lw=3)
