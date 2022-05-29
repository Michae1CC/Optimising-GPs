# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:21:43 2015
Fast Walsh-Hadamard Transform with Sequency Order
Author: Ding Luo@Fraunhofer IOSB
Edited by: Michael Ciccotosto-Camp
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def fwht_helper(X: np.array) -> np.array:
    """ Fast Walsh-Hadamard Transform
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3 algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications of Walsh and Related Functions.
    """
    rows, cols = X.shape
    output = np.empty((rows, cols))
    for i in range(rows):
        x = X[i, :].reshape(-1)
        N = x.size
        G = N >> 1  # Number of Groups
        M = 2  # Number of Members in Each Group
        # First stage
        y = np.zeros((G, 2))
        y[:, 0] = x[0::2] + x[1::2]
        y[:, 1] = x[0::2] - x[1::2]
        x = y
        # Second and further stage
        stages = int(np.log2(N))+1
        G_ = 0
        M_ = 0
        for _ in range(2, stages):
            G_ = G >> 1
            M_ = M << 1
            y = np.zeros((G_, M_))
            y[0:G_, 0:M_:4] = x[0:G:2, 0:M:2] + x[1:G:2, 0:M:2]
            y[0:G_, 1:M_:4] = x[0:G:2, 0:M:2] - x[1:G:2, 0:M:2]
            y[0:G_, 2:M_:4] = x[0:G:2, 1:M:2] - x[1:G:2, 1:M:2]
            y[0:G_, 3:M_:4] = x[0:G:2, 1:M:2] + x[1:G:2, 1:M:2]
            x = y
            G = G_
            M = M_
        output[i, :] = y[0, :] / float(N)
    return output


def fwht_primer(n: int, dtype=np.float64):
    """
    Primes the fwht_helper for a specific vector size
    """
    if not (n & (n-1) == 0):
        raise ValueError("The number of rows must be a of 2.")
    X = np.ones((n, n), dtype=dtype)
    fwht_helper(X)
    return


def fwht(X: np.array) -> np.array:
    """
    Computes the matrix product
        X H
    where H is the Hadamard matrix using the Fast Walsh-Hadamard transform, 
    similar to MatLab's implementation. The number of columns of
    X is assumed to be a power of two.
    """
    _, cols = X.shape
    if not (cols & (cols-1) == 0):
        raise ValueError("Number of columns must be a of 2.")
    return fwht_helper(X)


if __name__ == "__main__":
    # Example use
    n = 2**2
    x = (np.eye(n, n))
    wx = fwht(x)
    print(wx)
