import numpy as np
from pathlib import Path

# Read results
res_p = Path("results/lassobench/")

# Draw pictures
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

def plot_data(data, label):
    print(f"{label}: {data.shape}")
    ax.plot(np.minimum.accumulate(data, axis=1).mean(0), label=label)

def plotLasso(func_name, EM_DIM=10):
    if func_name == 'synt_high':
        DIM = 300
    elif func_name == 'DNA':
        DIM = 180
    else:
        raise NotImplementedError
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-turbo.npy")),
        label='TuRBO'
    )
    plot_data(
        data=np.load(res_p.joinpath(f"{func_name}-D{DIM}-turboD.npy")),
        label='TuRBO-D'
    )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-rembo.npy")),
    #     label="REMBO"
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-alebo.npy")),
    #     label="ALEBO"
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-sir_bo.npy")),
    #     label="SIR-BO"
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-sobol.npy")),
    #     label="Sobol"
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-kdr_bo.npy")),
    #     label="KDR-BO"
    # )
    # plot_data(
    #     np.load(res_p.joinpath(f"{func_name}-D{DIM}-d{EM_DIM}-mkdr_bo.npy"))
    #     label="MKDR-BO"
    # )
    ax.grid(True)
    ax.set_title(f"LassoBench-{func_name} (D = {DIM})", fontsize=18)
    ax.set_xlabel("Number of evaluations", fontsize=18)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=18)
    # ax.set_ylim([0, 30])
    ax.legend()
    # plt.show()
    plt.show()

if __name__ == '__main__':
    plotLasso("DNA")
    # plotLasso('synt_high')
## BO
# Y_hebo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-hebo.npy")), axis=0)
# Y_bo = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo.npy")), axis=0)
# Y_bo_warp = np.mean(np.load(res_p.joinpath(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-bo_warp.npy")), axis=0)
# Y_bo_moo = np.mean(np.load(res_p.joinpath(f"Top2-D{DIM}-moo.npy")), axis=0)
# Y_sobol = np.mean(np.load(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-sobol.npy"), axis=0)



# HDBO
# ax.plot(np.minimum.accumulate(Y_mkdr_bo), label="MKDR-BO")

# ax.plot(np.minimum.accumulate(Y_sir_bo), label="SIR-BO")

# BO
# ax.plot(np.minimum.accumulate(Y_hebo), label="HEBO")
# ax.plot(np.minimum.accumulate(Y_bo), label="BO")
# ax.plot(np.minimum.accumulate(Y_bo_warp), label="BO-Warp")
# ax.plot(np.minimum.accumulate(Y_bo_moo), label="BO-MOO")
# ax.plot(np.minimum.accumulate(Y_sobol), label="SOBOL")


# ax.plot([0, len(store_data)], [0, 0], "k--", lw=3)


