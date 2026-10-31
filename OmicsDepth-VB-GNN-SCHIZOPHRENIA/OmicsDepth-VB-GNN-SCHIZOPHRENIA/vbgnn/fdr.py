"""Dependency-aware FDR control of inferred edges (SI Section 1.2, Algorithm 1 l.15)."""
from __future__ import annotations

import numpy as np


def posterior_fdr_select(Pi: np.ndarray, alpha: float = 0.05, mask: np.ndarray | None = None):
    """Bayesian (posterior) FDR rule of SI Section 1.2.

    With ``l_ij = 1 - pi_hat_ij`` sorted increasingly, select the ``k*`` edges with
    the smallest ``l`` where ``k* = max{k : k^{-1} sum_{l<=k} l_(l) <= alpha}``.
    The posterior expected false discovery proportion of the selected set is then
    ``<= alpha`` irrespective of the dependence between edges.

    Returns ``(selected_mask, expected_fdp)``.
    """
    d = Pi.shape[0]
    mask = (1.0 - np.eye(d)) if mask is None else mask
    idx = np.flatnonzero(mask.ravel() > 0)
    ell = 1.0 - Pi.ravel()[idx]
    order = np.argsort(ell, kind="stable")
    running = np.cumsum(ell[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(running <= alpha)
    sel = np.zeros(d * d, dtype=bool)
    if len(ok) == 0:
        return sel.reshape(d, d), 0.0
    k = ok[-1] + 1
    sel[idx[order[:k]]] = True
    return sel.reshape(d, d), float(running[k - 1])


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05, dependence: str = "positive") -> np.ndarray:
    """Benjamini-Hochberg (PRDS) or Benjamini-Yekutieli (arbitrary dependence) step-up.

    ``dependence="arbitrary"`` replaces ``alpha`` by ``alpha / sum_{k=1}^m 1/k``.
    """
    p = np.asarray(pvals, float)
    shape = p.shape
    p = p.ravel()
    m = np.sum(np.isfinite(p))
    if dependence == "arbitrary":
        alpha = alpha / np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    thresh = alpha * np.arange(1, len(p) + 1) / m
    passed = np.flatnonzero(p[order] <= thresh)
    sel = np.zeros(len(p), dtype=bool)
    if len(passed):
        sel[order[: passed[-1] + 1]] = True
    return sel.reshape(shape)
