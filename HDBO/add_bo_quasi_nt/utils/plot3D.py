# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from torch import Tensor
import numpy as np
import torch


def plot_mll(mll: Callable, bounds: Tensor, num=100):
    bounds_np = bounds.numpy()
    x = np.linspace(bounds_np[0, 0], bounds_np[0, 1], num)
    y = np.linspace(bounds_np[1, 0], bounds_np[1, 1], num)

    xgrid, ygrid = np.meshgrid(x, y)
    xy = np.zeros(num**2)
    for i, xxyy in enumerate(zip(xgrid.flatten(), ygrid.flatten())):
        xy[i] = mll(torch.tensor(xxyy, dtype=bounds.dtype)).item()
    xy = xy.reshape(xgrid.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, xy, cmap='terrain')
    ax.set_xlabel('bandwidth')
    ax.set_ylabel('scale')
    ax.set_zlabel('mll')
    plt.show()


def plot_acq(acq: Callable, bounds: Tensor, num=100):
    bounds_np = bounds.numpy()
    x = np.linspace(bounds_np[0, 0], bounds_np[0, 1], num)
    y = np.linspace(bounds_np[1, 0], bounds_np[1, 1], num)

    xgrid, ygrid = np.meshgrid(x, y)
    xy = torch.tensor(list(zip(xgrid.flatten(), ygrid.flatten())), dtype=bounds.dtype)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, acq(xy).numpy().reshape(xgrid.shape), cmap='terrain')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('acq')
    plt.show()
