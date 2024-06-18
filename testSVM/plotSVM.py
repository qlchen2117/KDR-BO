import numpy as np
from pathlib import Path

# pick_data, DIM = "DNA", 180

pick_data, DIM = 'svm', 388

EM_DIM = 10
# Read results
res_p = Path("results/svm")

# Draw pictures
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))

## HDBO
store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-turbo.npy")), axis=0)
ax.plot(np.minimum.accumulate(store_data), label="TuRBO")
store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-rembo.npy")), axis=0)
ax.plot(np.minimum.accumulate(store_data), label="REMBO")
store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-alebo.npy")), axis=0)
ax.plot(np.minimum.accumulate(store_data), label="ALEBO")
store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-sir_bo.npy")), axis=0)
ax.plot(np.minimum.accumulate(store_data), label="SIR-BO")
store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-sobol.npy")), axis=0)
ax.plot(np.minimum.accumulate(store_data), label="Sobol")
# store_data = np.mean(np.load(res_p.joinpath(f"{pick_data}-D{DIM}-d{EM_DIM}-kdr_bo.npy")), axis=0)
# ax.plot(np.minimum.accumulate(store_data), label="KDR-BO")
# Y_mkdr_bo = np.mean(np.load(res_p.joinpath(f"lasso-dna-D{DIM}-d{EM_DIM}-mkdr_bo.npy")), axis=0)

# Y_sir_bo = np.mean(scio.loadmat(f"{type(func).__name__}-D{DIM}-d{EM_DIM}-SIR_BO.mat"), axis=1)


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

ax.grid(True)
ax.set_title(f"LassoBench-{pick_data} (D = {DIM})", fontsize=18)
ax.set_xlabel("Number of evaluations", fontsize=18)
# ax.set_xlim([0, len(Y_np)])
ax.set_ylabel("Best value found", fontsize=18)
# ax.set_ylim([0, 30])
ax.legend()
plt.show()
# plt.savefig(f"lasso-{pick_data}.pdf")
