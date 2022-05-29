#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import sys
import types
from typing import Callable, Mapping
import time
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from scipy.special import expit as sigmoid
from scipy.stats import bernoulli
from sklearn.metrics import accuracy_score, mean_squared_error

from data import load_data
from linalg import cg, minres
from rff import exact_kernel, create_rff, get_mem_usage
from nystrom import create_nystrom


def W_compute(a, inv=False):
    """Helper to compute matrix W for classification tasks."""
    r = sigmoid(a) * (1 - sigmoid(a))
    if inv:
        r = 1.0 / r
    return np.diag(r.ravel())


def gp_reg_pred(X_train: np.ndarray, Y_train: np.ndarray, x_pred: np.ndarray,
                sigma: float = None, kern_method: Callable = exact_kernel,
                kernel_args: dict = None, lin_solve: Callable = None,
                lin_solve_args: dict = None, meta: dict = None):
    """
    Makes real-values predictions using Gaussian processes.

    Parameters:
        X_train:
            An n-by-d np.ndarray of training inputs.
        Y_train:
            A n-by-1 np.ndarray of training labels corresponding to the 
            training inputs.
        x_pred:
            An 1-by-d np.ndarray of input to make a prediction at.
        sigma:
            The bandwidth parameters of the kernel matrix. Must be 
            greater than 0.
        kern_method:
            A callable which produces a Gram matrix when provided X_train
            as its first input.
        kernel_args:
            Additional key-word arguments provided to the kern_method when 
            called.
        lin_solve:
            A callable that solves the linear system Ax = b when invoked as
            lin_solve(A, b, x0) where x0 is the startign guess.
        lin_solve_args:
            Additional key-word arguments provided to the lin_solve when 
            called.
    Returns:
        Returns the prediction as a float.
    """
    n, d = X_train.shape
    lin_solve_args = lin_solve_args or {}
    kernel_args = kernel_args or {}
    if sigma is None:
        print("[WARN]: No value of sigma specified.")
        kernel_args["sigma"] = 1.0
    else:
        kernel_args["sigma"] = sigma
    # Create the Gram matrix corresponding to the training data set.
    x0 = np.zeros((n, 1))
    # noise variance.
    s = np.var(Y_train.squeeze())
    alpha = None
    if lin_solve is None and kern_method is exact_kernel:
        print("Usage exact")
        K = kern_method(X_train, **kernel_args)
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        alpha = np.linalg.solve(K + s * np.eye(n), Y_train)
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
    elif lin_solve is None:
        kernel_args["decomp"] = True
        Ks_inv = None
        dc = kern_method(X_train, **kernel_args)
        if type(dc) is tuple:
            U, C_inv = dc
            V = U.T
        else:
            V = dc
            U = V.T
            D, _ = V.shape
            C_inv = np.eye(D)

        A_inv = (1/s) * np.eye(n)
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        S = np.linalg.pinv(C_inv + (V @ A_inv @ U),
                           hermitian=True, rcond=10e-14)
        Ks_inv = A_inv - (1/(s * s)) * (U @ S @ V)
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
        alpha = np.matmul(Ks_inv, Y_train)
    elif lin_solve is not None and kern_method is exact_kernel:
        K = kern_method(X_train, **kernel_args)
        #######################################################################
        # TEST NORMAL NOISE
        # rm = np.random.normal(loc=0, scale=1e-1, size=(n, n))
        # rm = (rm + rm.T) / 2
        # K += rm
        #######################################################################
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        alpha, av_res = lin_solve(
            K + s * np.eye(n), Y_train, x0, **lin_solve_args)
        meta["res"] = av_res
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
    # else:
    #     kernel_args["decomp"] = True
    #     meta["mem"] = get_mem_usage()
    #     meta["time"] = time.perf_counter()
    #     Ks_inv = None
    #     dc = kern_method(X_train, **kernel_args)
    #     print("Got here")
    #     if type(dc) is tuple:
    #         U, C_inv = dc
    #         V = U.T
    #     else:
    #         V = dc
    #         U = V.T
    #         D, _ = V.shape
    #         C_inv = np.eye(D)
    #     A_inv = (1/s) * np.eye(n)
    #     S_inv = C_inv + (V @ A_inv @ U)
    #     SV, av_res = lin_solve(S_inv, V, **lin_solve_args)
    #     print("Solved lin sys")
    #     Ks_inv = A_inv - (1/(s * s)) * (U @ SV)
    #     meta["res"] = av_res
    #     meta["time"] = time.perf_counter() - meta["time"]
    #     meta["mem"] = get_mem_usage() - meta["mem"]
    #     alpha = np.matmul(Ks_inv, Y_train)
    else:
        # kernel_args["decomp"] = True
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        Ks_inv = None
        K = kern_method(X_train, **kernel_args)
        alpha, av_res = lin_solve(
            K + s * np.eye(n), Y_train, x0, **lin_solve_args)
        meta["res"] = av_res
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
        # alpha = np.matmul(Ks_inv, Y_train)
    # alpha = np.linalg.solve(K + s * np.eye(n), Y_train)
    K_ = exact_kernel(X_train, x_pred, sigma=sigma)
    Ef = K_.T @ alpha
    return Ef


def gp_bin_cls_pred(X_train: np.ndarray, Y_train: np.ndarray, X_pred: np.ndarray,
                    max_iter: int = 20, sigma: float = None, kern_method: Callable = exact_kernel,
                    kernel_args: dict = None, lin_solve: Callable = None,
                    lin_solve_args: dict = None, preds: bool = True, meta: dict = None):
    """
    Makes binary predictions using Gaussian processes.

    Parameters:
        X_train:
            An n-by-d np.ndarray of training inputs.
        Y_train:
            A n-by-1 np.ndarray of training labels corresponding to the 
            training inputs. Each label should take a binary 0 or 1 value.
        X_pred:
            An n_s-by-d np.ndarray of input to make a prediction at.
        sigma:
            The bandwidth parameters of the kernel matrix. Must be 
            greater than 0.
        kern_method:
            A callable which produces a Gram matrix when provided X_train
            as its first input.
        kernel_args:
            Additional key-word arguments provided to the kern_method when 
            called.
        lin_solve:
            A callable that solves the linear system Ax = b when invoked as
            lin_solve(A, b, x0) where x0 is the startign guess.
        lin_solve_args:
            Additional key-word arguments provided to the lin_solve when 
            called.
        preds:
            An argument to specify if the a vector of binary predictions 
            should be return or the probability that the class belongs to the 
            class of 1. Default is True.
    Returns:
        Returns the prediction as a float.
    """
    n, d = X_train.shape
    lin_solve_args = lin_solve_args or {}
    kernel_args = kernel_args or {}
    if sigma is None:
        print("[WARN]: No value of sigma specified.")
        kernel_args["sigma"] = 1.0
    else:
        kernel_args["sigma"] = sigma
    # a sufficiently small value to aid numerical computation
    mu = 1.0e-6
    kernel_args["decomp"] = False
    K_a = kern_method(X_train, **kernel_args)
    K_s = exact_kernel(X_train, X_pred, sigma=sigma)
    a_h = np.zeros_like(Y_train)
    I = np.eye(X_train.shape[0])
    for _ in range(max_iter):
        W = W_compute(a_h)
        # T = lin_solve(I + W @ K_a, K_a)
        T = np.linalg.solve(I + W @ K_a, K_a)
        a_h_new = T.dot(Y_train - sigmoid(a_h) + W.dot(a_h))
        a_h_diff = np.abs(a_h_new - a_h)
        a_h = a_h_new
        if not np.any(a_h_diff > 10e-10):
            break
    W_inv = W_compute(a_h, inv=True)
    R = None
    if lin_solve is None and kern_method is exact_kernel:
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        R = np.linalg.solve(W_inv + K_a, K_s)
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
    elif lin_solve is None:
        kernel_args["decomp"] = True
        Ks_inv = None
        dc = kern_method(X_train, **kernel_args)
        if type(dc) is tuple:
            U, C_inv = dc
            V = U.T
        else:
            V = dc
            U = V.T
            D, _ = V.shape
            C_inv = np.eye(D)

        A_inv = W_inv
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        S = np.linalg.pinv(C_inv + (V @ A_inv @ U),
                           hermitian=True, rcond=10e-14)
        Ks_inv = A_inv - A_inv @ (U @ S @ V) @ A_inv
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
        R = np.matmul(Ks_inv, Y_train)
    elif lin_solve is not None and kern_method is exact_kernel:
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        R, av_res = lin_solve(W_inv + K_a, K_s, X0=None, **lin_solve_args)
        meta["res"] = av_res
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
    else:
        kernel_args["decomp"] = False
        meta["mem"] = get_mem_usage()
        meta["time"] = time.perf_counter()
        K_a = kern_method(X_train, **kernel_args)
        R, av_res = lin_solve(W_inv + K_a, K_s, X0=None, **lin_solve_args)
        meta["res"] = av_res
        meta["time"] = time.perf_counter() - meta["time"]
        meta["mem"] = get_mem_usage() - meta["mem"]
    a_mu = K_s.T.dot(Y_train - sigmoid(a_h))
    a_var = (1 + mu) - np.sum(R * K_s, axis=0).reshape(-1, 1)
    kappa = 1.0 / np.sqrt(1.0 + np.pi * a_var / 8)
    # probability that the instance belongs to the binary class of 1
    probs = sigmoid(kappa * a_mu)
    if not preds:
        return probs
    probs[probs <= 0.5] = 0
    probs[~(probs <= 0.5)] = 1
    preds_vec = probs.astype(int)
    return preds_vec


def gp_mult_cls_pred(X_train: np.ndarray, Y_train: np.ndarray, X_pred: np.ndarray, **kwargs):
    """
    Makes binary predictions using Gaussian processes.

    Parameters:
        X_train:
            An n-by-d np.ndarray of training inputs.
        Y_train:
            A n-by-1 np.ndarray of training labels corresponding to the 
            training inputs. Each label should take an integer value 0-(C-1)
            where C is the number of classes.
        X_pred:
            An n_s-by-d np.ndarray of input to make a prediction at.
        kwargs:
            See binary prediction.
    Returns:
        Returns the prediction as a float.
    """
    n, d = X_train.shape
    kwargs["preds"] = False
    cs = np.unique(Y_train.astype(int).squeeze()).tolist()
    cls_mapping = dict(enumerate(cs))
    # print(dict((v, k) for (k, v) in cls_mapping.items()))
    Y_train = (np.vectorize((dict((v, k) for (k, v) in cls_mapping.items())).get)(
        Y_train)).astype(int).reshape(-1, 1)
    cls_probs = []
    # print(np.unique(Y_train))
    # print(cls_mapping.keys())
    call_meta = kwargs["meta"]
    pred_times = []
    pred_mems = []
    pred_ress = []
    for c in cls_mapping.keys():
        meta = {}
        kwargs["meta"] = meta
        b_train = Y_train.copy()
        b_train[Y_train == c] = 1
        b_train[Y_train != c] = 0
        b_prob_c = gp_bin_cls_pred(X_train, b_train, X_pred, **kwargs)
        pred_times.append(meta["time"])
        pred_mems.append(meta["mem"])
        pred_ress.append(meta["res"])
        cls_probs.append(b_prob_c.squeeze())
    call_meta["time"] = np.sum(np.array(pred_times))
    call_meta["mem"] = np.sum(np.array(pred_mems))
    call_meta["res"] = np.nanmean(np.array(pred_ress))
    cls_probs = np.column_stack(cls_probs)
    # print(cls_probs)
    preds = np.argmax(cls_probs, axis=1)
    preds = (np.vectorize(cls_mapping.get)(preds)).astype(int)
    return preds.reshape(-1, 1)


def test_reg():
    from math import pi
    from matplotlib import pyplot as plt
    np.random.seed(0)

    """
    n = 1000
    X_train = np.linspace(0, 2 * pi, num=n).reshape(-1, 1)
    f = np.sin(X_train)
    Y_train = f + (np.random.normal(size=(n, 1)) - 0.5) * 0.2
    kernel_args = {
        "s": 100,
        "replace": True,
        "method": "data_col"
    }
    # plt.plot(X_train, f, c='k')
    # plt.scatter(X_train, Y_train, c='r', s=2)
    # plt.title('True function')
    # plt.show()

    X_preds = np.linspace(0, (5/2) * pi, num=50).reshape(-1, 1)

    Y_preds = gp_reg_pred(X_train, Y_train, X_preds, sigma=5.0, lin_solve=minres,
                          lin_solve_args={"itnlim": 500, "reltol": 1e-12},
                          kern_method=create_nystrom,
                          kernel_args=kernel_args)

    plt.plot(X_preds.squeeze(), Y_preds.squeeze(), c='r')
    plt.scatter(X_train.squeeze(), Y_train.squeeze(), c='b', s=2)
    plt.title('Gaussian Process fitted')
    plt.show()

    """
    data, labels = load_data("abalone", labels=True)
    n, d = data.shape
    X_train = data[0:1000, :].reshape(-1, d)
    Y_train = labels[0:1000].reshape(-1, 1)
    x_pred = data[4001:4501, :]
    y_act = labels[4001:4501]
    kernel_args = {
        # "s": 500,
        # "k": 20,
        "D": 1500,
        # "replace": False,
        "method": "rff"
    }
    lin_solve_args = {
        "itnlim": 100,
        "reltol": 10e-10
    }
    y_pred = gp_reg_pred(X_train, Y_train, x_pred, sigma=1.0, lin_solve=minres, lin_solve_args=lin_solve_args,
                         kern_method=create_rff, kernel_args=kernel_args, meta={})
    print("test acc: ", mean_squared_error(y_act, y_pred))
    return


def test_bin_cls():
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from data import load_data
    np.random.seed(0)
    """
    X = np.arange(0, 5, 0.2).reshape(-1, 1)
    X_test = np.arange(-2, 7, 0.1).reshape(-1, 1)

    a = np.sin(X * np.pi * 0.5) * 2
    t = bernoulli.rvs(sigmoid(a))
    pt_test = gp_bin_cls_pred(X, t, X_test, sigma=1.0)

    plt.scatter(X[t == 1], t[t == 1], c='r')
    plt.scatter(X[t == 0], t[t == 0], c='b')
    plt.plot(X_test, pt_test, label='Prediction', color='green')
    plt.axhline(0.5, X_test.min(), X_test.max(),
                color='black', ls='--', lw=0.5)
    plt.title('Predicted class 1 probabilities')
    plt.xlabel('$x$')
    plt.ylabel('$p(t_*=1|\mathbf{t})$')
    plt.legend()
    plt.show()
    """
    data, labels = load_data("spam", labels=True)
    n, d = data.shape
    # print(labels.dtype)
    # print(n, d)
    train_indexes = np.hstack(
        [np.arange(0, 200, dtype=int), np.arange(n - 200, n, dtype=int)]).flatten()
    test_indexes = np.hstack(
        [np.arange(1000, 1100, dtype=int), np.arange(n - 1100, n - 1000, dtype=int)]).flatten()
    X_train = data[train_indexes, :].reshape(-1, d)
    Y_train = labels[train_indexes].reshape(-1, 1)
    # X_pred = data[test_indexes, :].reshape(-1, d)
    # Y_act = labels[test_indexes]
    # X_train, Y_train = load_iris(return_X_y=True)
    # update_index = Y_train == 2.0
    # X_train = X_train[~update_index]
    # Y_train = Y_train[~update_index]
    Y_train = Y_train.reshape(-1, 1)
    Y_act = Y_train
    X_pred = X_train
    kernel_args = {
        # "s": 400,
        # "k": 20,
        "D": 1500,
        # "replace": False,
        "method": "rff"
    }
    meta = {}
    Y_pred = gp_mult_cls_pred(X_train, Y_train, X_pred,
                              kern_method=create_rff, kernel_args=kernel_args,
                              sigma=1.0, preds=False, lin_solve=minres, lin_solve_args={"itnlim": 5}, meta=meta
                              )
    # print(test_indexes)
    # print(f"{test_indexes.shape}")
    # print(Y_act)
    # print(Y_pred)
    # print(Y_pred.shape)
    # Y_pred = np.random.rand(X_pred.shape[0], 1)
    # Y_pred[Y_pred <= 0.5] = 0
    # Y_pred[~(Y_pred <= 0.5)] = 1

    Y_pred = Y_pred.flatten()
    Y_act = Y_act.flatten().astype(int)
    print(f"{sum(Y_pred == Y_act)} / {int(X_pred.shape[0])}")
    from pprint import pprint
    pprint(meta)


def main():
    # test_reg()
    test_bin_cls()


if __name__ == "__main__":
    main()
