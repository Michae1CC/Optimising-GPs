#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

from typing import Callable

import numpy as np
from numba import jit


@jit(nopython=True)
def minres_vec(A: np.ndarray, b: np.ndarray, x0: np.ndarray, itnlim: int = -1, reltol: float = 1.0e-20) -> np.array:
    """
    Solves the linear system Ax = b using the minimum residual method. 
    Behaves similar to scipy.sparse.linalg.minres.

    Parameters:
        A:
            The real symmetric n-by-n matrix of the linear system.
        b:
            Right hand side of the linear system. Has shape (n,1).
        x0:
            Starting guess for the solution. Has shape (n,1).
        itnlim:
            Maximum number of iterations. Iteration will stop 
            after maxiter steps even if the specified tolerance has 
            not been achieved. Default is 5*n.
        reltol:
            Tolerance to achieve. The algorithm terminates when the 
            relative error is below reltol. Default is 1.0e-20.
    Returns:
        x:
            The converged solution.
    Sources:
        https://github.com/optimizers/nlpy/blob/master/nlpy/krylov/minres.py
    """

    n: int = b.shape[0]

    # Read keyword arguments
    if itnlim < 0:
        itnlim = 2*n
    shift: float = 0.0
    # itnlim: int = kwargs.get('itnlim', 5*n)
    # reltol: float = kwargs.get('reltol',   1.0e-12)

    # eps: float = np.finfo(float).eps
    eps: float = 2.220446049250313e-16
    istop: int = 0
    itn = 0
    Anorm: float = 0.0
    Acond: float = 0.0
    rnorm: float = 0.0
    ynorm: float = 0.0
    done: bool = False
    r1: np.ndarray = b
    y: np.ndarray = b.copy()
    beta1: float = np.sum(np.square(y))
    x: np.ndarray = x0.copy()
    #  If b = 0 exactly, stop with x = 0.
    if beta1 < 0.0:
        istop = 8
        done = True
    elif beta1 == 0.0:
        done = True
    else:
        beta1 = np.sqrt(beta1)
    oldb: float = 0.0
    beta: float = beta1
    dbar: float = 0.0
    epsln: float = 0.0
    qrnorm: float = beta1
    phibar: float = beta1
    rhs1: float = beta1
    Arnorm: float = 0.0
    rhs2: float = 0.0
    tnorm2: float = 0.0
    ynorm2: float = 0.0
    cs: float = -1.0
    sn: float = 0.0
    w: np.ndarray = np.zeros((n, 1))
    w2: np.ndarray = np.zeros((n, 1))
    r2: np.ndarray = r1.copy()

    if itnlim == 0:
        return x, (np.linalg.norm(A @ x - b) / np.linalg.norm(b))
    elif done:
        return x, float(rnorm)

    while itn < itnlim:
        itn = itn + 1
        # -------------------------------------------------------------
        # Obtain quantities for the next Lanczos vector vk+1, k=1,2,...
        # The general iteration is similar to the case k=1 with v0 = 0:
        #
        #   p1      = Operator * v1  -  beta1 * v0,
        #   alpha1  = v1'p1,
        #   q2      = p2  -  alpha1 * v1,
        #   beta2^2 = q2'q2,
        #   v2      = (1/beta2) q2.
        #
        # Again, y = betak P vk,  where  P = C**(-1).
        # .... more description needed.
        # -------------------------------------------------------------
        s = 1.0/beta                # Normalize previous vector (in y).
        v = s*y                     # v = vk if P = I
        y = np.dot(A, v)
        y -= shift*v
        if itn >= 2:
            y = y - (beta/oldb)*r1
        alfa: float = np.dot(v.T, y)[0][0]          # alphak
        y = (- alfa/beta)*r2 + y
        r1 = r2.copy()
        r2 = y.copy()
        oldb = beta               # oldb = betak
        beta = np.dot(r2.T, y)[0][0]          # beta = betak+1^2
        if beta < 0:
            istop = 6
            break
        beta = np.sqrt(beta)
        tnorm2 = tnorm2 + alfa*alfa + oldb*oldb + beta*beta
        if itn == 1:                  # Initialize a few things.
            if beta/beta1 <= 10*eps:  # beta2 = 0 or ~ 0.
                istop = -1            # Terminate later.
            gmax = abs(alfa)      # alpha1
            gmin = gmax             # alpha1
        # print("y shape ", y.shape)
        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
        oldeps = epsln
        delta = cs * dbar + sn * alfa  # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa  # gbar 1 = alfa1     gbar k
        epsln = sn * beta  # epsln2 = 0         epslnk+1
        dbar = -  cs * beta  # dbar 2 = beta2     dbar k+1
        root = np.sqrt(gbar*gbar + dbar*dbar)
        Arnorm = phibar * root
        gamma = np.sqrt(gbar*gbar + beta*beta)      # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1
        # Update  x.
        denom = 1.0/gamma
        w1 = w2.copy()
        w2 = w.copy()
        w = (v - oldeps*w1 - delta*w2) * denom
        x += phi*w  # x     = x  +  phi*w
        # Go round again.
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        ynorm2 = z*z + ynorm2
        rhs1 = rhs2 - delta*z
        rhs2 = -epsln*z
        # Estimate various norms and test for convergence.
        Anorm = np.sqrt(tnorm2)
        ynorm = np.sqrt(ynorm2)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * reltol
        diag = gbar
        if diag == 0:
            diag = epsa
        qrnorm = phibar
        rnorm = qrnorm
        test1 = rnorm / (Anorm*ynorm)  # ||r|| / (||A|| ||x||)
        test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)
        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q * H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.
        Acond = gmax/gmin
        # See if any of the stopping criteria are satisfied.
        # In rare cases istop is already -1 from above (Abar = const*I)
        # if istop == 0:
        #     t1 = 1 + test1      # These tests work if reltol < eps
        #     t2 = 1 + test2
        #     if t2 <= 1:
        #         istop = 2
        #     if t1 <= 1:
        #         istop = 1

        #     if itn >= itnlim:
        #         istop = 6
        #     if Acond >= 0.1/eps:
        #         istop = 4
        #     if epsx >= beta1:
        #         istop = 3
        #     if test2 <= reltol:
        #         istop = 2
        #     if test1 <= reltol:
        #         istop = 1
        # elif istop > 0:
        #     break
        # rnorm = np.linalg.norm(np.dot(A, x) - b)
        # my_r = np.linalg.norm(A @ x - b)
        # print("my res", np.linalg.norm(A @ x - b) / np.linalg.norm(b))
        # print("old res", rnorm / np.linalg.norm(b))
        if (rnorm <= reltol):
            break
    # print(itn)
    return x, float(rnorm) / np.linalg.norm(b)


@jit(nopython=True)
def cg_vec(A: np.ndarray, b: np.ndarray, x0: np.ndarray, itnlim: int = -1, reltol: float = 1.0e-10) -> np.array:
    """
    Solves the linear system Ax = b using the conjugate gradient method.
    Behaves similar to scipy.sparse.linalg.cg.

    Parameters:
        A:
            The real symmetric n-by-n matrix of the linear system. A must 
            be a hermitian, positive definite matrix.
        b:
            Right hand side of the linear system. Has shape (n,1).
        x0:
            Starting guess for the solution. Has shape (n,1).
        itnlim:
            Maximum number of iterations. Iteration will stop 
            after maxiter steps even if the specified tolerance has 
            not been achieved. Default is 2*n.
        reltol:
            Tolerance to achieve. The algorithm terminates when the 
            relative error is below reltol. Default is 1.0e-10.
    Returns:
        x:
            The converged solution.
    Sources:
        https://github.com/optimizers/nlpy/blob/master/nlpy/krylov/pcg.py
        https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/cg/cg.py
    """

    n: int = b.shape[0]

    if itnlim < 0:
        itnlim = 2*n

    b = b.copy().reshape(-1, 1)
    x = np.zeros((n, 1))
    r = b.copy()
    T = 0
    rel_res_best = np.inf
    rel_res = 1

    delta = r.T.dot(r)
    p = np.copy(r)
    x = x0.copy()

    if itnlim == 0:
        return x, (np.linalg.norm(A @ x - b) / np.linalg.norm(b))

    while T < itnlim and rel_res > reltol:
        T += 1
        Ap = A.dot(p)
        pAp = p.T.dot(Ap)
        pAp = pAp[0][0]
        # print(pAp)
        # pAp = np.maximum(pAp, 10e-15)
        # if pAp < 0:
        #     print('pAp =', pAp)
        #     raise ValueError('pAp < 0 in myCG')

        alpha = delta/pAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = np.linalg.norm(r)/np.linalg.norm(b)
        if rel_res_best > rel_res:
            rel_res_best = rel_res
        prev_delta = delta
        delta = r.T.dot(r)
        p = r + (delta/prev_delta)*p

    return x, float((np.linalg.norm(A @ x - b) / np.linalg.norm(b)))


def lin_solve(solver: Callable, A: np.ndarray, B: np.ndarray, X0: np.ndarray = None, **kwargs):
    """
    Solves the linear system Ax = b a provided solver.

    Parameters:
        solver:
            A callable, that solves the linear system Ax = b where A, b and x0
            have shapes (n,n), (n,1) and (n,1) respectively. It is provided
            these arguments in the order (A, b, x0, **kwargs).
        A:
            The real symmetric n-by-n matrix of the linear system.
        B:
            Right hand side of the linear system. Has shape (n,m).
        X0:
            Starting guess for the solution. Has shape (n,m).
    Returns:
        X:
            The converged solution, which shape (n,m).
    """
    if B.shape[1] == 1:
        return solver(A, B, X0, **kwargs)
    n, m = B.shape
    X = np.empty_like(B)
    res_l = []
    for i in range(m):
        x0 = None
        if X0 is None:
            x0 = np.zeros((n, 1))
        else:
            x0 = X0[:, i].reshape(-1, 1)
        b = B[:, i].reshape(-1, 1)
        X_temp, res = solver(A, b, x0, **kwargs)
        X[:, i] = X_temp.squeeze()
        res_l.append(res)
    return X, np.nanmean(np.array(res_l))


def minres(A: np.ndarray, B: np.ndarray, X0: np.ndarray = None, **kwargs):
    """
    Solves the linear system AX = B using the minimum residual method.
    See minres_vec and lin_solve for parameters and return value.
    """
    return lin_solve(minres_vec, A, B, X0, **kwargs)


def cg(A: np.ndarray, B: np.ndarray, X0: np.ndarray = None, **kwargs):
    """
    Solves the linear system AX = B using the conjugate gradient method.
    See cg_vec and lin_solve for parameters and return value.
    """
    return lin_solve(cg_vec, A, B, X0, **kwargs)


def prime_lin_solve(solver: Callable, n: int):
    """
    Primes the vector linear solver.
    """
    A: np.ndarray = np.eye(n)
    b: np.ndarray = np.ones((n, 1))
    x0: np.ndarray = np.zeros((n, 1))
    solver(A, b, x0, itnlim=2, reltol=1.0)
    return


def prime_cg(n: int):
    """
    Primes the Conjugate Gradient solver for a system of size n.
    """
    prime_lin_solve(cg_vec, n)
    return


def prime_minres(n: int):
    """
    Primes the Conjugate Gradient solver for a system of size n.
    """
    prime_lin_solve(minres_vec, n)
    return


def test_minres():
    from scipy.sparse.linalg import minres as sp_minres

    n = 500
    reltol = 1e-10
    A = (np.random.rand(n, n))
    A = np.dot(A, A.transpose())
    b = np.ones((n, 1))
    x0 = np.zeros((n, 1))

    itnlims = np.linspace(2, 500, 20, dtype=int)
    my_relerr = []
    sp_relerr = []

    for itnlim in itnlims:
        x = minres(A, b, x0, reltol=reltol, itnlim=itnlim)
        my_relerr.append(np.linalg.norm((A @ x) - b) / np.linalg.norm(x))

        x, _ = sp_minres(A, b, x0=x0, tol=reltol, maxiter=itnlim)
        sp_relerr.append(np.linalg.norm((A @ x) - b) / np.linalg.norm(x))

    import matplotlib.pyplot as plt
    plt.yscale("log")
    plt.plot(itnlims, my_relerr)
    plt.plot(itnlims, sp_relerr)
    plt.legend(["my func", "sp_func"])
    plt.show()


def test_cg():
    from scipy.sparse.linalg import cg as sp_cg

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html

    n = 250
    reltol = 1e-7
    A = (np.random.rand(n, n))
    A = np.dot(A, A.transpose())
    b = np.ones((n, 1))
    x0 = np.zeros((n, 1))

    itnlims = np.linspace(2, 500, 60, dtype=int)
    my_relerr = []
    sp_relerr = []

    for itnlim in itnlims:
        x = cg(A, b, x0, reltol=reltol, itnlim=itnlim)
        my_relerr.append(np.linalg.norm((A @ x) - b) / np.linalg.norm(x))

        x, _ = sp_cg(A, b, x0=x0, tol=reltol, maxiter=itnlim)
        sp_relerr.append(np.linalg.norm((A @ x) - b) / np.linalg.norm(x))

    import matplotlib.pyplot as plt
    plt.yscale("log")
    plt.plot(itnlims, my_relerr)
    plt.plot(itnlims, sp_relerr)
    plt.legend(["my func", "sp_func"])
    plt.show()


def test():
    import time

    n = 500
    A = (np.random.rand(n, n))
    A = np.dot(A, A.transpose())
    d = 1
    B = np.ones((n, d))
    x0 = np.zeros((n, d))
    reltol = 1e-9

    prime_cg(n)

    t0 = time.perf_counter()
    X = cg(A, B, x0, reltol=reltol)
    print(time.perf_counter() - t0)
    print(np.linalg.norm((A @ X) - B))


if __name__ == "__main__":
    test()
