"""Construction of the prior adjacency ``A_prior`` from curated pathway edges (e.g. KEGG)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prior_from_edgelist(edges: pd.DataFrame | str, feature_names: list[str],
                        source_col: str = "source", target_col: str = "target",
                        verbose: bool = True) -> np.ndarray:
    """``A_prior[i, j] = 1`` if ``feature_i -> feature_j`` is a curated edge.

    Edges whose endpoints are not among the retained features are ignored
    (reported when ``verbose``).
    """
    if isinstance(edges, str):
        edges = pd.read_csv(edges)
    pos = {f: i for i, f in enumerate(feature_names)}
    d = len(feature_names)
    A = np.zeros((d, d))
    skipped = 0
    for s, t in zip(edges[source_col], edges[target_col]):
        if s in pos and t in pos and s != t:
            A[pos[s], pos[t]] = 1.0
        else:
            skipped += 1
    if verbose and skipped:
        print(f"[prior] {skipped} curated edges skipped (endpoint not in feature set)")
    return A


def edgelist_from_matrix(M: np.ndarray, feature_names: list[str], mask: np.ndarray | None = None,
                         value_name: str = "weight") -> pd.DataFrame:
    rows = []
    d = len(feature_names)
    for i in range(d):
        for j in range(d):
            if i != j and (mask is None or mask[i, j]):
                rows.append((feature_names[i], feature_names[j], float(M[i, j])))
    return pd.DataFrame(rows, columns=["source", "target", value_name])
