import numpy as np


def sq_dist(aa, bb=None):
    # sq_dist - a function to compute a matrix of all pairwise squared distances
    # between two sets of vectors, stored in the columns of the two matrices, a
    # (of size D by n) and b (of size D by m). If only a single argument is given
    # or the second matrix is empty, the missing matrix is taken to be identical
    # to the first.

    # Usage: C = sq_dist(a, b)
    #     or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])

    # Where a is of size Dxn, b is of size Dxm (or empty), C is of size nxm.

    dim, num = aa.shape

    # Computation of a^2 - 2*a*b + b^2 is less stable than (a-b)^2 because numerical
    # precision can be lost when both a and b have very large absolute value and the
    # same sign. For that reason, we subtract the mean from the data beforehand to
    # stabilise the computations. This is OK because the squared error is
    # independent of the mean.
    if bb is None:  # subtract mean
        mu = np.mean(aa, axis=1, keepdims=True)
        aa = aa - mu
        bb = aa
        num2 = num
    else:
        dim2, num2 = bb.shape
        assert dim == dim2
        mu = (num2/(num+num2)) * np.mean(bb, axis=1, keepdims=True) + (num/(num+num2)) * np.mean(aa, axis=1, keepdims=True)
        aa = aa - mu
        bb = bb - mu

    # compute squared distances
    gram = np.sum(aa ** 2, axis=0, keepdims=True).T + np.sum(bb ** 2, axis=0, keepdims=True) - 2 * aa.T @ bb
    gram = np.maximum(gram, 0)  # numerical noise can cause C to negative i.e. C > -1e-14
    return gram


if __name__ == '__main__':
    xx = np.array([[1, 2], [3, 4], [5, 6]])
    print(sq_dist(xx))
    yy = np.array([[5, 6], [1, 2], [3, 4]]) * -1
    print(sq_dist(xx, yy))