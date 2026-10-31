"""Shared machinery for the synthetic benchmarks (main text Section 3).

``run_method`` maps a method name to a fitted binary adjacency so that the
simulation (Table 1) and ablation (Figure 2) scripts score every method in the
same way:

* ``VB-GNN``          -- this package, EGS regularizer, posterior-FDR selection
* ``VB-GNN-Shannon``  -- identical model with the Shannon regularizer (ablation)
* ``PC``, ``NOTEARS`` -- self-contained implementations in :mod:`baselines`
* ``DAG-GNN``, ``GraN-DAG``, ``DCDI`` -- original PyTorch code bases, optional
  (see :mod:`baselines.external`); raise :class:`baselines.MissingBaseline` when
  not installed and are then skipped by the scripts.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .fdr import posterior_fdr_select
from .metrics import edge_auroc, structure_metrics
from .model import VBGNN, VBGNNConfig
from .simulation import make_graph, simulate_anm

VBGNN_METHODS = {"VB-GNN": "egs", "VB-GNN-Shannon": "shannon", "VB-GNN-NoReg": "none"}


def run_method(name: str, X: np.ndarray, vbgnn_cfg: dict | None = None,
               method_cfg: dict | None = None, tag: str | None = None,
               fdr_alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit ``name`` on ``X`` and return ``(B_hat, scores)``.

    ``B_hat[i, j] = 1`` means ``i -> j``; ``scores`` are continuous edge scores
    (posterior probabilities for VB-GNN) used for AUROC, or ``None``.
    """
    method_cfg = dict(method_cfg or {})
    if name in VBGNN_METHODS:
        cfg = dict(vbgnn_cfg or {})
        cfg["regularizer"] = VBGNN_METHODS[name]
        model = VBGNN(VBGNNConfig(**cfg)).fit(X)
        P = model.edge_probs
        sel, _ = posterior_fdr_select(P, fdr_alpha)
        return sel.astype(float), P
    if name == "PC":
        from baselines import pc
        return pc(X, **method_cfg), None
    if name == "NOTEARS":
        from baselines import notears_linear
        W = notears_linear(X, **method_cfg)
        return (W != 0).astype(float), np.abs(W)
    from baselines import EXTERNAL
    if name not in EXTERNAL:
        raise KeyError(f"unknown method {name!r}")
    if name == "DCDI":
        return EXTERNAL[name](X, tag=tag, **method_cfg), None
    return EXTERNAL[name](X, **method_cfg), None


def simulate_trial(graph: str, d: int, n: int, seed: int, rho: float = 0.0,
                   sigma2: float = 1.0, families=("gp", "mlp", "periodic")):
    """One Monte-Carlo replicate: ``(B_true, X)`` from a seeded generator."""
    rng = np.random.default_rng(seed)
    B = make_graph(graph, d, rng)
    X = simulate_anm(B, n, rng, rho=rho, sigma2=sigma2, families=tuple(families))
    return B, X


def evaluate(job: dict) -> list[dict]:
    """Worker: simulate one replicate and score every requested method on it.

    ``job`` keys: graph, d, n, seed, trial, rho, sigma2, methods, vbgnn, baselines,
    fdr_alpha, families. Returns one record per method (skipped methods carry
    ``status="missing"``).
    """
    B, X = simulate_trial(job["graph"], job["d"], job["n"], job["seed"],
                          job.get("rho", 0.0), job.get("sigma2", 1.0),
                          job.get("families", ("gp", "mlp", "periodic")))
    tag = f"{job['graph']}_trial{job['trial']}"
    out = []
    for name in job["methods"]:
        rec = {k: job[k] for k in ("graph", "d", "n", "trial", "seed")}
        rec.update(rho=job.get("rho", 0.0), sigma2=job.get("sigma2", 1.0), method=name,
                   true_edges=int(B.sum()))
        t0 = time.time()
        try:
            B_hat, scores = run_method(name, X, job.get("vbgnn"),
                                       job.get("baselines", {}).get(name), tag,
                                       job.get("fdr_alpha", 0.05))
        except Exception as e:  # missing optional baseline or solver failure
            from baselines import MissingBaseline
            rec.update(status="missing" if isinstance(e, MissingBaseline) else "error",
                       message=str(e)[:200])
            out.append(rec)
            continue
        rec.update(structure_metrics(B, B_hat))
        rec["auroc"] = edge_auroc(B, scores) if scores is not None else float("nan")
        rec.update(status="ok", seconds=round(time.time() - t0, 2))
        out.append(rec)
    return out


def summarize(df, by=("graph", "method"), metrics=("shd", "fdr", "tpr", "auroc")):
    """Mean and standard deviation of each metric over Monte-Carlo trials."""
    import pandas as pd

    ok = df[df["status"] == "ok"]
    agg = ok.groupby(list(by))[list(metrics)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index() if isinstance(agg, pd.DataFrame) else agg


def default_vbgnn_config() -> dict:
    return asdict(VBGNNConfig())
