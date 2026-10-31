"""Causal information flux and counterfactual do-interventions on a fitted VB-GNN."""
from __future__ import annotations

import numpy as np


def causal_flux(model, X: np.ndarray, groups: np.ndarray | None = None,
                Z: np.ndarray | None = None, batch: int = 256) -> dict:
    """Feature-wise causal information flux ``F_g`` (Algorithm 1, line 16).

    ``[F_g]_j = || d GNN(X_g) / d x_{g,j} ||_F`` -- the Frobenius norm of the
    Jacobian of the GNN output with respect to the j-th input column, over the
    samples of group ``g`` (e.g. a cell type). Returns ``{group: F_g}``.
    """
    X = np.asarray(X, float)
    groups = np.zeros(len(X), dtype=object) if groups is None else np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        Xg = X[groups == g]
        acc = np.zeros(X.shape[1])
        for s in range(0, len(Xg), batch):
            J = model.jacobian(Xg[s:s + batch], Z)          # (n, k, j)
            acc += np.sum(J * J, axis=(0, 1))
        out[g] = np.sqrt(acc)
    return out


def counterfactual_do(model, X: np.ndarray, interventions: dict[int, float | np.ndarray],
                      Z: np.ndarray | None = None, damping: float = 0.5,
                      max_iter: int = 500, tol: float = 1e-6, abduct: bool = True):
    """Counterfactual ``do(x_s = v)`` evaluated at the equilibrium of the fitted SEM.

    Because the graph may contain feedback loops, the intervened system is solved
    by damped fixed-point iteration ``x <- (1-w) x + w (mu(x) + eps)`` with the
    intervened coordinates clamped (rather than a single forward pass).
    With ``abduct=True`` each sample keeps its own exogenous residual
    ``eps = x - mu(x)`` (Pearl's abduction-action-prediction for additive noise).

    Returns ``(X_cf, info)`` with ``info = {"converged", "iterations", "residual"}``.
    """
    X = np.asarray(X, float)
    eps = X - model.predict(X, Z) if abduct else 0.0
    x = X.copy()
    for j, v in interventions.items():
        x[:, j] = v
    res = np.inf
    for it in range(1, max_iter + 1):
        new = model.predict(x, Z) + eps
        for j, v in interventions.items():
            new[:, j] = v
        nxt = (1.0 - damping) * x + damping * new
        res = float(np.max(np.abs(nxt - x)))
        x = nxt
        if res < tol:
            break
    return x, {"converged": bool(res < tol), "iterations": it, "residual": res}


def targeted_flux(model, X: np.ndarray, groups: np.ndarray, source: int,
                  targets: list[int], delta: float = 0.5, Z: np.ndarray | None = None,
                  **kw) -> dict:
    """Total (equilibrium) effect of a shift in ``source`` on ``targets`` per group.

    ``flux_g = mean_n || x_targets(do(source = x_s + delta)) - x_targets || / delta``
    -- the sensitivity of downstream responses to a targeted change of the source
    (used to compare cell types, main text Section 5 Step 4).
    """
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        Xg = X[groups == g]
        base, _ = counterfactual_do(model, Xg, {}, Z, **kw)
        pert, _ = counterfactual_do(model, Xg, {source: Xg[:, source] + delta}, Z, **kw)
        diff = (pert - base)[:, targets]
        out[g] = float(np.mean(np.linalg.norm(diff, axis=1)) / delta)
    return out


def fit_group_models(model, X: np.ndarray, groups: np.ndarray, A_prior: np.ndarray | None = None,
                     C: np.ndarray | None = None, iters: int = 1500, **overrides) -> dict:
    """Cell-type-specific parameter spaces: one VB-GNN per group, warm-started from
    the pooled fit ``model`` and trained further on that group's samples only.

    A single pooled model shares its structural functions across groups, so group
    differences in its Jacobian reflect only where each group sits on the shared
    non-linearity, not group-specific effect sizes. Refitting per group lets the
    slopes (and edge posteriors) differ, which is what the cell-type flux of the
    main text (Section 5, Step 4) compares. Returns ``{group: fitted VBGNN}``.
    """
    from dataclasses import asdict

    from .model import VBGNN, VBGNNConfig

    groups = np.asarray(groups)
    cfg = asdict(model.config)
    cfg.update(iters=iters, adv_warmup=min(cfg.get("adv_warmup", 0), iters // 4), **overrides)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        gm = VBGNN(VBGNNConfig(**cfg))
        gm.fit(X[m], A_prior=A_prior, C=None if C is None else C[m], Sigma=model.Sigma,
               feature_names=model.feature_names, init=model.params)
        out[g] = gm
    return out
