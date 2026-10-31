"""Causal Match Score drug repurposing and preranked GSEA."""
from __future__ import annotations

import numpy as np


def disease_flux_vector(X: np.ndarray, diagnosis: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Cell-type-specific disease causal-flux vector.

    ``v_j = F_{g,j} * (mean_case_j - mean_control_j)``: the disease shift of each
    feature, weighted by how strongly that feature carries causal flux in group g.
    """
    X = np.asarray(X, float)
    dx = np.asarray(diagnosis).astype(bool)
    return flux * (X[dx].mean(0) - X[~dx].mean(0))


def causal_match_score(signature: np.ndarray, disease_vector: np.ndarray) -> float:
    """Normalised directional alignment (cosine) between a compound signature and
    the disease causal-flux vector on shared features. Strongly negative values
    mean the compound pushes expression *against* the disease trajectory.
    """
    s = np.asarray(signature, float)
    v = np.asarray(disease_vector, float)
    ok = np.isfinite(s) & np.isfinite(v)
    s, v = s[ok], v[ok]
    den = np.linalg.norm(s) * np.linalg.norm(v)
    return float(s @ v / den) if den > 0 else float("nan")


def preranked_gsea(scores: dict[str, float], gene_sets: dict[str, list[str]],
                   weight: float = 1.0, n_perm: int = 1000, seed: int = 0,
                   min_size: int = 2) -> list[dict]:
    """Preranked GSEA (Subramanian et al. 2005) with gene-set permutation.

    Returns a list of ``{"set", "es", "nes", "pval", "size"}`` sorted by NES.
    """
    rng = np.random.default_rng(seed)
    genes = np.array(list(scores))
    vals = np.array([scores[g] for g in genes], float)
    order = np.argsort(-vals)
    genes, vals = genes[order], vals[order]
    N = len(genes)
    pos = {g: i for i, g in enumerate(genes)}

    def es_of(hit_idx: np.ndarray) -> float:
        hit = np.zeros(N, bool)
        hit[hit_idx] = True
        w = np.abs(vals) ** weight
        nr = w[hit].sum()
        if nr == 0:
            return 0.0
        p_hit = np.cumsum(np.where(hit, w / nr, 0.0))
        p_miss = np.cumsum(np.where(~hit, 1.0 / (N - hit.sum()), 0.0))
        dev = p_hit - p_miss
        return float(dev[np.argmax(np.abs(dev))])

    out = []
    for name, members in gene_sets.items():
        idx = np.array([pos[g] for g in members if g in pos], int)
        if len(idx) < min_size or len(idx) >= N:
            continue
        es = es_of(idx)
        null = np.array([es_of(rng.choice(N, len(idx), replace=False)) for _ in range(n_perm)])
        same = null[np.sign(null) == np.sign(es)] if es != 0 else null
        mean_null = np.mean(np.abs(same)) if len(same) else np.nan
        nes = es / mean_null if mean_null and np.isfinite(mean_null) else np.nan
        pval = (np.sum(np.abs(same) >= abs(es)) + 1) / (len(same) + 1)
        out.append({"set": name, "es": es, "nes": float(nes), "pval": float(pval), "size": int(len(idx))})
    return sorted(out, key=lambda r: -np.nan_to_num(r["nes"], nan=-np.inf))
