import torch


def dist2gp(x, c):
    """ Calculates squared distance between two sets of points.
    Args:
        x: A 'n_d x ndims' tensor
        c: A 'n_c x ndims' tensor
    Returns:
        A 'n_d x n_c' tensor
    """
    n_d, dim_x = x.shape
    n_c, dim_c = c.shape
    if dim_x != dim_c:
        raise Exception('Data dimension does not match dimension of centers.')
    xx = torch.sum(x ** 2, axis=1)  # (n_d,)
    cc = torch.sum(c ** 2, axis=1)  # (n_c,)
    xc = x @ c.T  # (n_data, n_centres)
    return torch.repeat_interleave(xx.unsqueeze(-1), n_c, dim=1) + torch.repeat_interleave(cc.unsqueeze(0), n_d, dim=0) - 2. * xc


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
    print(dist2gp(x, x))
