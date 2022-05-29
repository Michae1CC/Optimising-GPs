#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import sys
import types
import inspect
import tracemalloc
from typing import Mapping
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from time import perf_counter


def is_prob_func(object):
    return isinstance(object, types.FunctionType) and object.__module__ == __name__ and ("_prob" in str(object))


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
    return np.exp(-pairwise_sq_dists / sigma**2)


def best_rank_k(X: np.ndarray, k: int) -> np.ndarray:
    """
    Provides the best rank k approximation for the matrix X.
    """
    # Perform SVD on im and obtain individual matrices
    P, D, Q = np.linalg.svd(X, full_matrices=True)
    # Select top "rank" singular values
    return np.matrix(P[:, :k]) * np.diag(D[:k]) * np.matrix(Q[:k, :])


def gram_sls_prob(X: np.ndarray, sigma: float, k: int = 20, A: np.ndarray = None) -> np.ndarray:
    """
    Computes the best rank statistical leverage scores of the data matrix X 
    using the exact Gram matrix A. For details see 
    "Revisiting the Nystrom method" pg 2.
    """
    if A is None:
        A = exact_kernel(X, sigma=sigma)
    # Get the left unitary matrix of the eigen value decomposition
    _, VL, _ = sp.linalg.eig(A, left=True)
    VL = VL.real
    # Compute the squared Euclidean norm of each row
    VL = np.square(VL[:, :k]) / k
    p = np.sum(VL, axis=1)
    return p / np.sum(p)


def data_sls_prob(X: np.ndarray, sigma: float, k: int = 20, A: np.ndarray = None) -> np.ndarray:
    """
    Computes the best rank statistical leverage scores only using the data 
    matrix.
    """
    # Get the QR decomposition
    Q, _ = np.linalg.qr(X)
    # Compute the squared Euclidean norm of each row
    Q = np.square(Q)
    p = np.sum(Q, axis=1)
    return p / np.sum(p)


def gram_col_prob(X: np.ndarray, sigma: float, k: int = 20, A: np.ndarray = None) -> np.ndarray:
    """
    Computes the column probabilites of a data matrix using the exact Gram
    matrix A. For details see
    "On the Nystrom Method for Approximating a Gram Matrix" pg 2160
    """
    if A is None:
        A = exact_kernel(X, sigma=sigma)
    exact_sq = np.square(A)
    exact_f = np.sum(exact_sq)
    # Sum over the columns
    # see: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    p = np.sum(exact_sq, axis=1).squeeze() / exact_f
    return p / np.sum(p)


def data_col_prob(X: np.ndarray, sigma: float, k: int = 20, A: np.ndarray = None) -> np.ndarray:
    """
    Computes the column probabilites of a data matrix using only the data 
    matrix.
    """
    exact_sq = np.square(X)
    exact_f = np.sum(exact_sq)
    # Sum over the columns
    # see: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    p = np.sum(exact_sq, axis=1).squeeze() / exact_f
    return p / np.sum(p)


def rls_prob(X: np.ndarray, sigma: float, k: int = 20, A: np.ndarray = None) -> np.ndarray:
    """
    Computes the best rank-k ridge leverages scores of a Gram matrix A 
    (computed from the data matrix X) and normalizes them to form a 
    probability distribution. For details see
    "Randomized Numerical linear Algebra" pg 89, and
    "Input Sparsity Time Low-Rank Approximation" pg 6
    """
    n, d = X.shape
    # Need to calculate the optimal lambda for the ridge leverage scores.
    # Refer to "Input Sparsity Time Low-Rank Approximation" pg 6.
    if A is None:
        A = exact_kernel(X, sigma=sigma)
    A_k = best_rank_k(A, k)
    lambda_ = (np.linalg.norm(A - A_k, ord="fro") ** 2) / k
    A_T = A.T
    h_tau_X = np.linalg.solve((A_T + lambda_ * np.eye(n)), A_T).T
    nu_eff = np.trace(h_tau_X)
    p = np.diag(h_tau_X)
    p = p / nu_eff
    p[p < 0] = 0
    return p / np.sum(p)


def create_uni(X: np.ndarray, s: int, sigma: float, A: np.ndarray = None, replace: bool = False, meta: Mapping = None) -> np.ndarray:
    """
    Creates a Nystrom approximation for the uniform sketching method.
    See:
    "Randomized Numerical Linear Algebra" pg 63
    "On the Nystrom Method for Approximating a Gram Matrix" pg 2160
    """
    if A is None:
        A = exact_kernel(X, sigma=sigma)
    n, d = X.shape
    basis_inds = np.random.choice(
        n, size=s, replace=replace, p=(np.ones(n) / n))
    S = np.eye(n, n)[:, basis_inds]
    am0 = get_mem_usage()
    st0 = perf_counter()
    Y = A @ S
    nu = np.sqrt(n) * np.finfo(np.float64).eps
    Y = Y + nu * S
    W = np.linalg.pinv(S.T @ Y, hermitian=True, rcond=10e-3)
    N = np.matmul(np.matmul(Y, W), Y.T)
    if meta is not None:
        meta["sketch_time"] = perf_counter() - st0
        meta["prob_time"] = 0
        meta["add_mem"] = get_mem_usage() - am0

    return N


def create_probs(X: np.ndarray, A: np.ndarray, method: str, sigma: float = 1, k: int = 20):
    """
    Computes the probability distrbution for a given method.
    """

    n, _ = X.shape

    if "uni" in method:
        return (np.ones(n) / n)

    method += "_prob"
    # Get a mapping of all the probability function names to references
    prob_func_map = dict(inspect.getmembers(
        sys.modules[__name__], predicate=is_prob_func))
    if method not in prob_func_map:
        raise ValueError("Method " + str(method) + " not supported.")
    prob_func = prob_func_map[method]
    p = prob_func(X, sigma, k, A)
    return p


def create_nystrom(X: np.ndarray, s: int = 100, A: np.ndarray = None, p: np.ndarray = None, k: int = 20,
                   sigma: float = 1, replace: bool = False, method: str = "uni",
                   decomp: bool = False, meta: Mapping = None) -> np.ndarray:
    """
    Creates a Nystrom approximation for the Gram matrix associated with the
    data matrix X.

    Parameters:
        X:
            The data matrix for which the approximated Gram matrix will be 
            computed for.
        s:
            The number of randomly samples to be used in the Nystrom 
            approximation.
        A:
            The exact kernel of the data matrix X. This will be used for 
            calculating probabilites for various methods.
        k:
            A sampling parameter used for different methods.
        sigma:
            The variance parameter of the RBF kernel.
        replace:
            Whether or not samples for the Nystrom approximation are selected
            with/without replacement.
        method:
            The method (abbr string format) used to determine sampling 
            probabilites.
        decomp:
            Return the matrix in its decomposed form (i.e. return the
            matrices C and Wp).
        meta:
            A mapping used to capture performace information.
    See:
    "Randomized Numerical Linear Algebra" pg 63
    "On the Nystrom Method for Approximating a Gram Matrix" pg 2161
    """
    n, d = X.shape
    st0 = 0
    pt0 = 0
    method = method.lower().strip()

    if A is None:
        A = exact_kernel(X, sigma=sigma)

    if "uni" in method:
        N = create_uni(X, s, sigma, A, replace=replace, meta=meta)
        return N
    else:
        if p is None:
            pt0 = perf_counter()
            p = create_probs(X, A, method, sigma=sigma, k=k)
            if meta is not None:
                meta["prob_time"] = perf_counter() - pt0

    basis_inds = np.random.choice(
        n, size=s, replace=replace, p=p)
    S = np.eye(n, n)[:, basis_inds]
    am0 = get_mem_usage()
    st0 = perf_counter()
    cp = np.sqrt(s * p[basis_inds])
    C = (A @ S)
    C = C / np.sqrt(s * cp)
    nu = np.sqrt(n) * np.finfo(np.float64).eps
    C = C + nu * S
    W = S.T @ (A @ S)
    W = W / (s * np.sqrt(cp.reshape(-1, 1) @ cp.reshape(1, -1)))

    if decomp:
        if meta is not None:
            meta["sketch_time"] = perf_counter() - st0
            meta["add_mem"] = get_mem_usage() - am0
        return C, W

    Wp = np.linalg.pinv(W, hermitian=True, rcond=10e-14)
    N = np.matmul(np.matmul(C, Wp), C.T)
    if meta is not None:
        meta["sketch_time"] = perf_counter() - st0
        meta["add_mem"] = get_mem_usage() - am0

    return N


def main():

    from data import load_data
    data, labels = load_data("3D_spatial_network", labels=True)
    n, d = data.shape
    sigma = 1.0
    k = 40
    s = 1500

    p_rls = create_probs(None, None, "rls", sigma=sigma, k=k,
                         data_set="3D_spatial_network")
    k_exact = exact_kernel(data, sigma=sigma)
    p_uni = np.ones(n) / n
    k_uni_1 = create_nystrom(data, s, A=k_exact, k=k,
                             sigma=sigma, replace=True, p=p_rls)
    rel_error_1 = np.linalg.norm(
        k_exact - k_uni_1, ord="fro") / np.linalg.norm(k_exact, ord="fro")
    print(rel_error_1)


if __name__ == "__main__":
    main()
