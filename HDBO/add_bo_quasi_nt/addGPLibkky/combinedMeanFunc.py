from torch import Tensor
from typing import Callable, List


def combined_mean_func(x: Tensor, y: Tensor, common_mean_func: Callable, mean_funcs: List[Callable], decomposition: List[Tensor]):
    """
    The total mean function (obtained by adding the common and individual mean function
    Args:
        x:
        common_mean_func:
        mean_funcs:
        decomposition: list with elements
    Returns:
        mu0: (num,)
    """
    mu0 = common_mean_func(x, y)
    num_groups = len(decomposition)
    for k in range(num_groups):
        coord = decomposition[k]
        mu0 += mean_funcs[k](x[:, coord])
    return mu0
