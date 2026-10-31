"""Linear NOTEARS (Zheng et al., NeurIPS 2018), augmented-Lagrangian solver.

Self-contained NumPy/SciPy port of the reference algorithm (``notears_linear``):
least-squares loss, L1 penalty, hard acyclicity ``h(W) = Tr(exp(W o W)) - d = 0``.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt


def notears_linear(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100,
                   h_tol: float = 1e-8, rho_max: float = 1e16, w_threshold: float = 0.3) -> np.ndarray:
    n, d = X.shape
    X = X - X.mean(0)

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _loss(W):
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):
        E = sla.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G + lambda1, -G + lambda1), axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W
