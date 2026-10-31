"""Adversarial surrogate-variable module (SI Eq. 4-5, Assumption A3).

The discriminator ``D_psi(z_causal, z_nuisance)`` outputs the probability that a
pair was drawn from the joint posterior ``Q`` rather than from the product of
marginals ``T``. Samples from ``T`` are obtained by pairing ``z_causal`` and
``z_nuisance`` of *different* samples (row shuffling). The objective is::

    L_adv(phi, psi) = E_Q[ln D] + E_T[ln(1 - D)]

maximised over ``psi`` and minimised over the VB-GNN parameters, which drives
``I(Z_causal, Z_nuisance) -> 0`` (Jensen-Shannon optimum). The discriminator is
kept ``L_D``-Lipschitz by spectral normalisation of every weight matrix and its
output is confined to ``[eta, 1 - eta]`` (Assumption A3).
"""
from __future__ import annotations

import numpy as np

from .relaxation import sigmoid


def surrogate_variables(X: np.ndarray, n_sv: int, primary: np.ndarray | None = None,
                        covariates: np.ndarray | None = None) -> np.ndarray:
    """Surrogate variables in the spirit of SVA: leading principal components of
    ``X`` after removing the primary variable of interest (e.g. diagnosis) and the
    measured covariates.

    These are meant to capture *unmeasured* technical heterogeneity and are passed
    to the adversary as additional nuisance coordinates.

    .. warning::
       When the regulatory programme of interest is itself a dominant axis of
       variation (as in the demo cohort), the leading components *are* that
       programme and treating them as nuisance removes the signal to be
       discovered. Use ``n_sv > 0`` only when unmeasured technical factors are
       expected, and check that the components do not load on the features of
       interest.
    """
    if n_sv <= 0:
        return np.zeros((X.shape[0], 0))
    R = X - X.mean(0)
    cols = [np.asarray(a, float).reshape(len(X), -1) for a in (primary, covariates) if a is not None]
    if cols:
        P = np.column_stack([np.ones(len(X)), *cols])
        coef, *_ = np.linalg.lstsq(P, X, rcond=None)
        R = X - P @ coef
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    sv = U[:, :n_sv] * S[:n_sv]
    return (sv - sv.mean(0)) / (sv.std(0) + 1e-12)


class Discriminator:
    """Two-layer spectrally normalised MLP with output margin ``eta``."""

    def __init__(self, in_dim: int, hidden: int = 32, eta: float = 0.05,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.eta = eta
        self.params = {
            "W1": rng.normal(0, 1.0 / np.sqrt(in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1)),
            "b2": np.zeros(1),
        }
        self.spectral_normalize()

    def spectral_normalize(self) -> None:
        for k in ("W1", "w2"):
            s = np.linalg.norm(self.params[k], 2)
            if s > 1.0:
                self.params[k] /= s

    def forward(self, Zin: np.ndarray):
        p = self.params
        h = np.tanh(Zin @ p["W1"] + p["b1"])
        o = (h @ p["w2"] + p["b2"])[:, 0]
        s = sigmoid(o)
        D = self.eta + (1.0 - 2.0 * self.eta) * s
        return D, (Zin, h, s)

    def backward(self, dD: np.ndarray, cache) -> tuple[dict, np.ndarray]:
        Zin, h, s = cache
        p = self.params
        do = dD * (1.0 - 2.0 * self.eta) * s * (1.0 - s)
        g = {"w2": h.T @ do[:, None], "b2": np.array([do.sum()])}
        dh = do[:, None] @ p["w2"].T
        da = dh * (1.0 - h * h)
        g["W1"] = Zin.T @ da
        g["b1"] = da.sum(0)
        dZin = da @ p["W1"].T
        return g, dZin

    def objective(self, zc: np.ndarray, zn: np.ndarray, perm: np.ndarray):
        """``L_adv`` together with gradients w.r.t. ``psi`` and w.r.t. ``zc``.

        Returns ``(L_adv, grads_psi, dL/dzc)``; the discriminator *ascends*
        ``grads_psi`` while VB-GNN *descends* ``dL/dzc``.
        """
        nq = len(zc)
        ZQ = np.hstack([zc, zn])
        ZT = np.hstack([zc, zn[perm]])
        DQ, cQ = self.forward(ZQ)
        DT, cT = self.forward(ZT)
        L = float(np.mean(np.log(DQ)) + np.mean(np.log(1.0 - DT)))
        gQ, dZQ = self.backward(1.0 / DQ / nq, cQ)
        gT, dZT = self.backward(-1.0 / (1.0 - DT) / nq, cT)
        grads = {k: gQ[k] + gT[k] for k in gQ}
        dc = zc.shape[1]
        dzc = dZQ[:, :dc] + dZT[:, :dc]
        return L, grads, dzc
