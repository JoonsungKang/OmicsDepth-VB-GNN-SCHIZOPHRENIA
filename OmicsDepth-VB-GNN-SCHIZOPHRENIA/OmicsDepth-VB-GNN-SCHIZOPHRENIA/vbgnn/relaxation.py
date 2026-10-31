"""Binary Concrete / Gumbel-Softmax relaxation of the latent graph (SI Eq. 9)."""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def sample_logistic_noise(shape, rng: np.random.Generator) -> np.ndarray:
    """``L = g1 - g2`` with ``g1, g2 ~ Gumbel(0, 1)`` i.i.d. (standard logistic)."""
    return rng.gumbel(size=shape) - rng.gumbel(size=shape)


def concrete_sample(Pi: np.ndarray, tau: float, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relaxed adjacency ``Z_hat = Sigmoid((logit(pi) + L) / tau)`` and ``dZ_hat/dpi``.

    The derivative is ``Sigmoid'(u) / (tau * pi * (1 - pi))`` (proof of
    Proposition 1); for ``tau <= 1`` it is bounded by ``C_tau = 4/tau exp(|L|/tau)``.
    ``P(Z_hat > 1/2) = pi`` for every ``tau`` (rounding property).
    """
    u = (logit(Pi) + noise) / tau
    Z = sigmoid(u)
    dZ = Z * (1.0 - Z) / (tau * Pi * (1.0 - Pi))
    return Z, dZ
