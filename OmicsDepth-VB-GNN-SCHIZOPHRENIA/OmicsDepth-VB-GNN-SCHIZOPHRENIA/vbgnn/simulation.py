"""Synthetic benchmark data (main text Section 3.1).

* Graphs: Erdos-Renyi ER-k (expected ``m = k d`` edges) and scale-free SF-k
  (Barabasi-Albert preferential attachment, ``~ k d`` edges), oriented along a
  random topological order (DAG).
* Non-linear additive noise model ``X_i = f_i(X_pa(i)) + eps_i`` with ``f_i``
  drawn from Gaussian-process (squared-exponential kernel, random Fourier
  features), MLP (sigmoid / GELU) or periodic families.
* Correlated noise ``eps ~ N(0, Sigma)``, ``Sigma = sigma^2 [(1 - rho) I + rho 1 1^T]``.
"""
from __future__ import annotations

import numpy as np


def _orient(und: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = und.shape[0]
    perm = rng.permutation(d)
    rank = np.empty(d, int)
    rank[perm] = np.arange(d)
    B = np.zeros((d, d))
    for i, j in np.argwhere(np.triu(und, 1)):
        if rank[i] < rank[j]:
            B[i, j] = 1
        else:
            B[j, i] = 1
    return B


def erdos_renyi_dag(d: int, k: float, rng: np.random.Generator) -> np.ndarray:
    """ER-k DAG with expected ``k * d`` edges."""
    p = min(1.0, 2.0 * k / (d - 1))
    und = np.triu(rng.uniform(size=(d, d)) < p, 1).astype(int)
    return _orient(und + und.T, rng)


def scale_free_dag(d: int, k: float, rng: np.random.Generator) -> np.ndarray:
    """SF-k DAG via Barabasi-Albert preferential attachment (``~ k * d`` edges).

    Each new node attaches to ``k`` existing nodes chosen proportionally to degree;
    edges point from earlier (hub) nodes to later nodes, then nodes are relabelled.
    """
    k = int(round(k))
    d0 = k + 1
    und = np.zeros((d, d), int)
    und[:d0, :d0] = 1 - np.eye(d0, dtype=int)
    deg = und.sum(0).astype(float)
    for v in range(d0, d):
        p = deg[:v] / deg[:v].sum()
        targets = rng.choice(v, size=min(k, v), replace=False, p=p)
        und[v, targets] = und[targets, v] = 1
        deg[targets] += 1
        deg[v] += len(targets)
    # orient old -> new (hubs are upstream regulators), then permute labels
    B = np.triu(und, 1).astype(float)
    perm = rng.permutation(d)
    return B[np.ix_(perm, perm)]


def make_graph(kind: str, d: int, rng: np.random.Generator) -> np.ndarray:
    """``kind`` in {"ER-2", "ER-4", "SF-2", "SF-4", ...}."""
    fam, k = kind.split("-")
    if fam.upper() == "ER":
        return erdos_renyi_dag(d, float(k), rng)
    if fam.upper() == "SF":
        return scale_free_dag(d, float(k), rng)
    raise ValueError(kind)


def topological_order(B: np.ndarray) -> list[int]:
    d = B.shape[0]
    indeg = B.sum(0).astype(int).copy()
    order, stack = [], [i for i in range(d) if indeg[i] == 0]
    while stack:
        v = stack.pop()
        order.append(v)
        for w in np.flatnonzero(B[v]):
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    if len(order) != d:
        raise ValueError("graph is not acyclic")
    return order


class _RandomFunction:
    def __init__(self, n_in: int, family: str, rng: np.random.Generator):
        self.family = family
        if family == "gp":                       # SE-kernel GP via random Fourier features
            self.nf = 50
            self.ls = rng.uniform(0.8, 1.5)
            self.W = rng.normal(0, 1 / self.ls, (n_in, self.nf))
            self.b = rng.uniform(0, 2 * np.pi, self.nf)
            self.a = rng.normal(0, np.sqrt(2.0 / self.nf), self.nf)
        elif family in ("mlp", "mlp-sigmoid", "mlp-gelu"):
            self.act = rng.choice(["sigmoid", "gelu"]) if family == "mlp" else family.split("-")[1]
            h = 10
            self.W1 = rng.uniform(0.5, 2.0, (n_in, h)) * rng.choice([-1, 1], (n_in, h))
            self.W2 = rng.uniform(0.5, 2.0, h) * rng.choice([-1, 1], h)
        elif family == "periodic":
            self.w = rng.uniform(0.5, 1.5, n_in) * rng.choice([-1, 1], n_in)
            self.phase = rng.uniform(0, 2 * np.pi)
            self.lin = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
        else:
            raise ValueError(family)

    def __call__(self, Xpa: np.ndarray) -> np.ndarray:
        if self.family == "gp":
            return np.sqrt(2.0) * np.cos(Xpa @ self.W + self.b) @ self.a * 2.0
        if self.family.startswith("mlp"):
            A = Xpa @ self.W1
            if self.act == "sigmoid":
                H = 1 / (1 + np.exp(-A))
            else:
                H = 0.5 * A * (1 + np.tanh(np.sqrt(2 / np.pi) * (A + 0.044715 * A ** 3)))
            return H @ self.W2 / np.sqrt(self.W1.shape[1])
        s = Xpa @ self.w
        return np.sin(s + self.phase) + self.lin * s


def correlated_noise_cov(d: int, rho: float, sigma2: float = 1.0) -> np.ndarray:
    return sigma2 * ((1.0 - rho) * np.eye(d) + rho * np.ones((d, d)))


def simulate_anm(B: np.ndarray, n: int, rng: np.random.Generator, rho: float = 0.0,
                 sigma2: float = 1.0, families: tuple[str, ...] = ("gp", "mlp", "periodic"),
                 standardize: bool = True) -> np.ndarray:
    """Sample ``n`` observations from the non-linear ANM over DAG ``B`` (B[i, j] = 1: i -> j)."""
    d = B.shape[0]
    Sigma = correlated_noise_cov(d, rho, sigma2)
    E = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    X = np.zeros((n, d))
    for j in topological_order(B):
        pa = np.flatnonzero(B[:, j])
        if len(pa):
            f = _RandomFunction(len(pa), str(rng.choice(families)), rng)
            fx = f(X[:, pa])
            fx = (fx - fx.mean()) / (fx.std() + 1e-12)
            X[:, j] = fx + E[:, j]
        else:
            X[:, j] = E[:, j]
    if standardize:
        X = (X - X.mean(0)) / X.std(0)
    return X
