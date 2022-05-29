#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import tracemalloc
from math import pi, sqrt
from time import perf_counter
from typing import Mapping

import numpy as np
from scipy.spatial.distance import cdist

from fwht import fwht, fwht_primer


def get_mem_usage(key_type='filename') -> float:
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    total = sum(stat.size for stat in top_stats)
    return float(total / 1024)


def exact_kernel(X: np.ndarray, Y: np.ndarray = None, sigma: float = 1, **kwargs) -> np.ndarray:
    """
    Produces the exact kernel of a matrix using a RBF kernel. If only the first
    matrix is provided then the following Gram matrix is computed:

    K_{i,j} = k(x_i, x_j)

    where x_i represents the ith row. If a value for the Y matrix is given 
    then the following Gram matrix is computed:

    K_{i,j} = k(x_i, y_j).

    The value of sigma is the variance value required for the RBF kernel.
    """
    Y = X if Y is None else Y
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / (sigma**2))


def orf_rf_matrix(x: np.ndarray, D: int, sigma: int = 1) -> np.ndarray:
    """
    Computes the transformation matrix of the Orthogonal Random Features matrix.
    See:
    "Orthogonal Random Features" pg 3
    "Random Features for Kernel Approximation" pg 8
    """
    # n is the number of samples and d is the dimension of the samples from
    # our data set
    n, d = x.shape
    # Create the random features
    G = np.random.normal(size=(D, d))
    Q, _ = np.linalg.qr(G)
    s = np.sqrt(np.random.chisquare(D, size=(D, 1)))
    Q = s.reshape(-1, 1) * Q
    Q = Q / sigma
    Q = Q[:D, :d].conj().T

    return Q


def sorf_rf_matrix(x: np.ndarray, D: int, sigma: int = 1) -> np.ndarray:
    """
    Computes the transformation matrix of the Structured Orthogonal Random 
    Features matrix.
    See:
    "Orthogonal Random Features" pg 6
    "Random Features for Kernel Approximation" pg 8
    """
    # n is the number of samples and d is the dimension of the samples from
    # our data set
    n, d = x.shape
    # Get the next power of 2 for D
    n2 = 1 if D == 0 else (D - 1).bit_length()
    n2p2 = 1 << n2
    # HD1 = fwht(np.diag(2 * np.random.binomial(1, 0.5, n2p2) - 1))
    # HD2 = fwht(np.diag(2 * np.random.binomial(1, 0.5, n2p2) - 1))
    # HD3 = fwht(np.diag(2 * np.random.binomial(1, 0.5, n2p2) - 1))
    # HD1 = (np.sqrt(n2p2)) * HD1
    # HD2 = (np.sqrt(n2p2)) * HD2
    # HD3 = (np.sqrt(n2p2)) * HD3
    # W = np.dot(np.dot(HD1, HD2), HD3)
    # W = ((np.sqrt(n2p2) / sigma) * W[:D, :d]).conj().T
    sqrtn2p2 = np.sqrt(n2p2)
    W = fwht(np.diag(2 * np.random.binomial(1, 0.5, n2p2) - 1)[:d, :])
    W = (2 * np.random.binomial(1, 0.5, n2p2) - 1) * W
    W = fwht(W)
    W = (2 * np.random.binomial(1, 0.5, n2p2) - 1) * W
    W = fwht(W)
    W = (((sqrtn2p2 / sigma) * (sqrtn2p2 * sqrtn2p2 * sqrtn2p2))
         * W[:d, :D]).conj()
    return W


def rf_matrix(X: np.ndarray, D: int, sigma: int = 1, method: str = "rff") -> np.ndarray:
    """
    Creates the transformation matrix for a given method.
    """
    # n is the number of samples and d is the dimension of the samples from
    # our data set
    n, d = X.shape
    if method == "rff":
        return np.random.normal(size=(d, D)) / (sigma)
    elif method == "orf":
        return orf_rf_matrix(X, D=D, sigma=sigma)
    elif method == "sorf":
        return sorf_rf_matrix(X, D=D, sigma=sigma)

    raise ValueError("Method " + str(method) + " not supported.")


def create_rff(X: np.ndarray, sigma: int = 1.0, kkm: int = 5,
               D: int = None, method: str = "rff", decomp: bool = False,
               meta: Mapping = None) -> np.ndarray:
    """
    Creates an approximation of the gaussian kernel matrix 
    using the random features technique for the input data array.

    Parameters:
        X:
            The data matrix for which the approximated Gram matrix will be 
            computed for.
        sigma:
            The variance parameter of the RBF kernel.
        kkm:
            Used to compute the number of samples to use in the Monte Carlo
            estimate as #samples = kkm * d, where d is the number of 
            attributes of the data set. The parameters kkm or D should be
            provided but not both.
        D:
            The number of samples to use in the Monte Carlo estimate. The 
            parameters kkm or D should be provided but not both.
        method:
            The method (abbr string format) used to determine which 
            transformation matrix to use.
        decomp:
            Return the matrix in its decomposed form (i.e. just return the
            explicit feature map, Z).
    """
    if method not in {"rff", "orf", "sorf"}:
        raise ValueError("Method " + str(method) + " not supported.")
    # n is the number of samples and d is the dimension of the samples from
    # our data set
    n, d = X.shape
    # Create the number of samples to generate
    D = D or kkm * d
    am0 = get_mem_usage()
    t0 = perf_counter()
    sigma /= sqrt(2)
    W = rf_matrix(X, D=D, sigma=sigma, method=method)
    t0 = perf_counter() - t0
    t1 = perf_counter()
    # Create the random fourier features
    # b = 2 * pi * np.random.rand(1, D)
    # WXb = np.matmul(W.T, X.conj().T) + b.conj().T
    # Z = (sqrt(2 / D)) * np.cos(WXb)
    WX = np.matmul(W.T, X.conj().T)
    Z = (1 / sqrt(D)) * np.vstack([
        np.cos(WX),
        np.sin(WX)
    ])
    if decomp:
        if meta is not None:
            meta["rfm_time"] = perf_counter() - t1
            meta["trans_mat_time"] = t0
            meta["add_mem"] = get_mem_usage() - am0
        return Z
    ret = np.matmul(Z.T, Z)
    if meta is not None:
        meta["rfm_time"] = perf_counter() - t1
        meta["trans_mat_time"] = t0
        meta["add_mem"] = get_mem_usage() - am0
    return ret


def test_rff():

    from data import load_data
    data, labels = load_data("3D_spatial_network", labels=True)
    n, d = data.shape
    D = 8 * d
    print("Running")
    sigma = 10
    K_exact = exact_kernel(data, sigma=sigma)
    m0 = get_mem_usage()
    t1 = perf_counter()
    K_rff = create_rff(data, sigma=sigma, D=D, method="orf")
    print("Runtime: ", perf_counter() - t1)
    print("Mem Usage: ", get_mem_usage() - m0)
    print(np.linalg.norm(K_exact - K_rff, ord="fro") /
          np.linalg.norm(K_exact, ord="fro"))


def main():
    test_rff()


if __name__ == "__main__":
    main()
