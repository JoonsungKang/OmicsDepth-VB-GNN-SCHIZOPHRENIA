"""Peter-Clark (PC) algorithm with Fisher-z partial-correlation tests.

Stable PC skeleton search, v-structure orientation and Meek rules R1-R3. The
output CPDAG is returned as a directed adjacency ``B`` in which undirected edges
appear in both directions (``B[i, j] = B[j, i] = 1``). If ``causal-learn`` is
installed, :func:`pc_causallearn` wraps its implementation instead.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import norm


def _fisher_z(C: np.ndarray, i: int, j: int, S: tuple, n: int) -> float:
    idx = [i, j, *S]
    sub = C[np.ix_(idx, idx)]
    try:
        P = np.linalg.pinv(sub)
    except np.linalg.LinAlgError:
        return 1.0
    r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]) + 1e-300)
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - len(S) - 3, 1))
    return 2 * (1 - norm.cdf(abs(z)))


def pc(X: np.ndarray, alpha: float = 0.01, max_cond: int = 3) -> np.ndarray:
    n, d = X.shape
    C = np.corrcoef(X, rowvar=False)
    adj = ~np.eye(d, dtype=bool)
    sepset: dict = {}
    level = 0
    while level <= max_cond:
        snapshot = adj.copy()
        cont = False
        for i in range(d):
            for j in range(d):
                if i == j or not adj[i, j]:
                    continue
                nb = [k for k in np.flatnonzero(snapshot[i]) if k != j]
                if len(nb) < level:
                    continue
                cont = True
                for S in combinations(nb, level):
                    if _fisher_z(C, i, j, S, n) > alpha:
                        adj[i, j] = adj[j, i] = False
                        sepset[(i, j)] = sepset[(j, i)] = set(S)
                        break
        if not cont:
            break
        level += 1
    # orientation: G[i,j]=1 & G[j,i]=1 undirected ; G[i,j]=1 & G[j,i]=0 means i->j
    G = adj.astype(int)
    for k in range(d):
        nb = np.flatnonzero(adj[k])
        for i, j in combinations(nb, 2):
            if not adj[i, j] and k not in sepset.get((i, j), set()):
                G[k, i] = G[k, j] = 0       # i -> k <- j
    changed = True
    while changed:
        changed = False
        for i in range(d):
            for j in range(d):
                if not (G[i, j] and G[j, i]):
                    continue
                # R1: a -> i - j, a not adj j  =>  i -> j
                if any(G[a, i] and not G[i, a] and not adj[a, j] for a in range(d) if a != j):
                    G[j, i] = 0; changed = True; continue
                # R2: i -> k -> j and i - j  =>  i -> j
                if any(G[i, k] and not G[k, i] and G[k, j] and not G[j, k] for k in range(d)):
                    G[j, i] = 0; changed = True; continue
                # R3: i - k1 -> j, i - k2 -> j, k1 not adj k2  =>  i -> j
                ks = [k for k in range(d) if G[i, k] and G[k, i] and G[k, j] and not G[j, k]]
                if any(not adj[a, b] for a, b in combinations(ks, 2)):
                    G[j, i] = 0; changed = True
    return G.astype(float)


def pc_causallearn(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:  # pragma: no cover
    from causallearn.search.ConstraintBased.PC import pc as cl_pc

    cg = cl_pc(X, alpha, "fisherz", show_progress=False)
    G = cg.G.graph  # G[j,i]=1 & G[i,j]=-1 : i -> j ; both -1 : undirected
    d = G.shape[0]
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if G[j, i] == 1 and G[i, j] == -1:
                B[i, j] = 1
            elif G[i, j] == -1 and G[j, i] == -1:
                B[i, j] = B[j, i] = 1
    return B
