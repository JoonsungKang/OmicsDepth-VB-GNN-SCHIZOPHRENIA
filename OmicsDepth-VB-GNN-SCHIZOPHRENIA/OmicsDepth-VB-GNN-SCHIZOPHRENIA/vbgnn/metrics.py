"""Structure-recovery metrics: SHD, FDR, TPR, AUROC and directional AUROC."""
from __future__ import annotations

import numpy as np


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties counted 1/2); no scikit-learn dependency."""
    from scipy.stats import rankdata

    labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def roc_curve(scores: np.ndarray, labels: np.ndarray):
    """Return ``(fpr, tpr)`` arrays for plotting."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels, bool)[order]
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.concatenate([[0], np.cumsum(~y)])
    return fp / max(1, (~y).sum()), tp / max(1, y.sum())


def structure_metrics(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """Directed-graph metrics in the NOTEARS convention.

    * ``shd``  = extra + missing + reversed edges (a reversed edge counts once)
    * ``fdr``  = (reversed + false positive) / predicted
    * ``tpr``  = correctly oriented true positives / true edges
    """
    B_true = (np.asarray(B_true) != 0).astype(int)
    B_est = (np.asarray(B_est) != 0).astype(int)
    np.fill_diagonal(B_true, 0)
    np.fill_diagonal(B_est, 0)
    pred = np.argwhere(B_est)
    cond = set(map(tuple, np.argwhere(B_true)))
    cond_rev = {(j, i) for i, j in cond}
    tp = sum((i, j) in cond for i, j in pred)
    rev = sum((i, j) in cond_rev and (i, j) not in cond for i, j in pred)
    fp = len(pred) - tp - rev
    # SHD on the skeleton+orientation
    sk_true = np.triu(B_true + B_true.T) > 0
    sk_est = np.triu(B_est + B_est.T) > 0
    extra = int(np.sum(sk_est & ~sk_true))
    missing = int(np.sum(sk_true & ~sk_est))
    shd = extra + missing + rev
    return {
        "shd": int(shd),
        "fdr": float((rev + fp) / max(len(pred), 1)),
        "tpr": float(tp / max(len(cond), 1)),
        "nnz": int(len(pred)),
        "reversed": int(rev),
    }


def edge_auroc(B_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of ranking all ``d(d-1)`` ordered pairs by ``scores`` against ``B_true``."""
    d = B_true.shape[0]
    off = ~np.eye(d, dtype=bool)
    return _auc(scores[off], (B_true != 0)[off])


def directional_auroc(scores: np.ndarray, reference: list[tuple[int, int]],
                      n_random_negatives: int | None = None, seed: int = 0):
    """AUROC for directional predictions against a reference set of relations.

    Positives are the reference edges ``i -> j``; negatives are their reversals
    ``j -> i`` (wrong direction) plus, optionally, random ordered pairs that are
    neither a reference edge nor its reversal. Returns ``(auc, fpr, tpr)``.
    """
    rng = np.random.default_rng(seed)
    d = scores.shape[0]
    pos = list(reference)
    ref = set(pos) | {(j, i) for i, j in pos}
    neg = [(j, i) for i, j in pos if (j, i) not in set(pos)]
    if n_random_negatives:
        cand = [(i, j) for i in range(d) for j in range(d) if i != j and (i, j) not in ref]
        pick = rng.choice(len(cand), min(n_random_negatives, len(cand)), replace=False)
        neg += [cand[k] for k in pick]
    s = np.array([scores[i, j] for i, j in pos + neg])
    y = np.array([1] * len(pos) + [0] * len(neg), bool)
    fpr, tpr = roc_curve(s, y)
    return _auc(s, y), fpr, tpr
