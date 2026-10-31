"""Structural regularizers of VB-GNN (Supplementary Information, Sections 1.2, 2.2 and 3).

All functions act on the matrix of variational edge probabilities
``Pi = [pi_ij] in [0, 1]^{d x d}`` and return values together with *analytic*
gradients so that they can be used inside the NumPy optimiser and checked
against finite differences in ``tests/``.

Entropy-based Gini-Simpson (EGS) structural loss (SI Eq. 15)::

    L_struct = lambda * beta * D(Pi) * T_Sigma(Pi)
    D(Pi)       = 1 - ||Pi||_F^2 / a          (structural diversity index)
    T_Sigma(Pi) = Tr(Pi^T Sigma Pi) / b       (dependency coupling)

``a = b = 1`` reproduces the formula of the SI verbatim (``normalize=False``).
On the full hypercube that literal form can become negative (``D`` ranges over
``[1 - d^2, 1]``), so by default the Frobenius term of ``D`` is divided by the
number of candidate edges ``m = d(d-1)`` (``normalize=True``: ``a = m, b = 1``).
This keeps ``D in [0, 1]`` and ``L_struct >= 0`` while preserving every property
proved in the SI: constant curvature of ``D`` (now ``-2/m``), a uniformly bounded
Hessian (Theorem 1) and permutation invariance (Theorem 2). The coupling term
``T_Sigma`` is left unscaled so that the per-edge gradient
``2 lambda beta D [Sigma Pi]_ij`` does not vanish as ``d`` grows: it penalises an
edge ``i -> j`` in proportion to how strongly ``i`` co-varies with the other
putative parents of ``j`` (redundant, correlated parents). The constants of the
bounds are rescaled accordingly (see :func:`egs_hessian_bound`).
"""
from __future__ import annotations

import numpy as np


def offdiag_mask(d: int) -> np.ndarray:
    """Mask of the ``m = d(d-1)`` candidate edges (self-loops excluded)."""
    return 1.0 - np.eye(d)


def _scales(d: int, normalize: bool) -> tuple[float, float]:
    m = float(d * (d - 1)) if d > 1 else 1.0
    return (m, 1.0) if normalize else (1.0, 1.0)


def structural_diversity(Pi: np.ndarray, normalize: bool = True) -> float:
    """Gini-Simpson structural diversity index ``D = 1 - ||Pi||_F^2 / a``."""
    a, _ = _scales(Pi.shape[0], normalize)
    return float(1.0 - np.sum(Pi * Pi) / a)


def egs_struct_loss(Pi: np.ndarray, Sigma: np.ndarray, lam: float, beta: float,
                    normalize: bool = True) -> tuple[float, np.ndarray]:
    """EGS structural loss and its gradient (SI Eq. 15 and Eq. 19).

    Gradient::

        dL/dpi_ij = lambda*beta * ( -2 pi_ij / a * T + D * 2 [Sigma Pi]_ij / b )
    """
    a, b = _scales(Pi.shape[0], normalize)
    SP = Sigma @ Pi
    T = float(np.sum(Pi * SP)) / b            # Tr(Pi^T Sigma Pi) / b
    D = 1.0 - float(np.sum(Pi * Pi)) / a
    loss = lam * beta * D * T
    grad = lam * beta * ((-2.0 / a) * Pi * T + D * (2.0 / b) * SP)
    return loss, grad


def egs_struct_hessian(Pi: np.ndarray, Sigma: np.ndarray, lam: float, beta: float,
                       normalize: bool = True) -> np.ndarray:
    """Full ``d^2 x d^2`` Hessian of :func:`egs_struct_loss` w.r.t. ``vec(Pi)``.

    Matrix form of SI Eq. (32) (column-major ``vec``)::

        H = lambda*beta * [ -2 T/a I - 4/(ab) (p s^T + s p^T) + 2 D/b (I_d kron Sigma) ]

    with ``p = vec(Pi)`` and ``s = vec(Sigma Pi)``. Intended for small ``d``
    (verification of Theorem 1).
    """
    d = Pi.shape[0]
    a, b = _scales(d, normalize)
    p = Pi.reshape(-1, order="F")
    s = (Sigma @ Pi).reshape(-1, order="F")
    T = float(p @ s) / b
    D = 1.0 - float(p @ p) / a
    eye = np.eye(d * d)
    H = (-2.0 * T / a) * eye - (4.0 / (a * b)) * (np.outer(p, s) + np.outer(s, p)) \
        + (2.0 * D / b) * np.kron(np.eye(d), Sigma)
    return lam * beta * H


def egs_hessian_bound(Sigma: np.ndarray, lam: float, beta: float, d: int,
                      normalize: bool = True) -> float:
    """Uniform bound on ``||Hess L_struct||_2`` over ``[0,1]^{d x d}`` (Theorem 1(i)).

    For ``a = b = 1`` this is ``12 lambda beta lambda_max(Sigma) d^2``. In general,
    with ``||p||^2 <= d^2`` and ``|D| <= max(1, d^2/a - 1)`` the three terms of
    Eq. (32) give ``lambda beta lambda_max (2 d^2/(ab) + 8 d^2/(ab) + 2 Dmax/b)``.
    """
    lmax = float(np.linalg.eigvalsh(Sigma)[-1])
    a, b = _scales(d, normalize)
    dmax = max(1.0, d * d / a - 1.0)
    return lam * beta * lmax * (2.0 * d * d / (a * b) + 8.0 * d * d / (a * b) + 2.0 * dmax / b)


def egs_gradient_bound(Sigma: np.ndarray, lam: float, beta: float, d: int,
                       normalize: bool = True) -> float:
    """Uniform bound ``M_struct`` on ``|dL_struct/dpi_ij|`` (Proposition 1).

    For ``a = b = 1`` this is ``2 lambda beta lambda_max(Sigma) (d^2 + d^{5/2})``.
    """
    lmax = float(np.linalg.eigvalsh(Sigma)[-1])
    a, b = _scales(d, normalize)
    dmax = max(1.0, d * d / a - 1.0)
    return 2.0 * lam * beta * lmax * (d * d / (a * b) + dmax * np.sqrt(d) / b)


def shannon_struct_loss(Pi: np.ndarray, lam: float, normalize: bool = False,
                        eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """Shannon (binary-entropy) structural regularizer used by the baseline (SI Eq. 10-12).

    ``L = lambda * H_Sh(Pi)`` (SI Eq. 11) with
    ``H_Sh = -sum[pi ln pi + (1-pi) ln(1-pi)]`` and gradient
    ``lambda * (ln(1-pi) - ln pi)``, which diverges as ``pi -> 0`` or ``1``.
    ``normalize=True`` divides by ``m = d(d-1)``.
    """
    a, _ = _scales(Pi.shape[0], normalize)
    P = np.clip(Pi, eps, 1.0 - eps)
    H = -float(np.sum(P * np.log(P) + (1.0 - P) * np.log(1.0 - P)))
    grad = (lam / a) * (np.log(1.0 - P) - np.log(P))
    return lam * H / a, grad


def shannon_curvature(Pi: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """``|d^2 H_Sh / d pi_ij^2| = 1 / (pi (1 - pi))`` -- unbounded on the hypercube (Thm 1(iii))."""
    P = np.clip(Pi, eps, 1.0 - 1e-16)
    return 1.0 / (P * (1.0 - P))


def two_cycle_penalty(Pi: np.ndarray) -> tuple[float, np.ndarray]:
    """Expected number of 2-cycles ``h_2 = sum_{i<j} pi_ij pi_ji`` and gradient ``Pi^T``.

    Soft acyclicity prior ``p(Z) propto exp(-gamma h_2)`` (default in VB-GNN). It
    discourages the symmetric ``i <-> j`` duplicates that a cyclic pseudo-likelihood
    otherwise tends to produce, while leaving longer feedback loops (length >= 3,
    e.g. Succinate -> H3K27ac -> GRIN2A -> Mito-Stress -> Succinate) free. Its
    gradient is bounded by 1 entry-wise, so Proposition 1 is unaffected.
    """
    off = Pi * (1.0 - np.eye(Pi.shape[0]))
    return 0.5 * float(np.sum(off * off.T)), off.T.copy()


def acyclicity(Pi: np.ndarray, scale: float | None = None) -> tuple[float, np.ndarray]:
    """NOTEARS characteristic ``h(A) = Tr(exp(A o A)) - d`` and gradient.

    VB-GNN uses ``h`` only as a *soft* (probabilistic) prior term with a
    user-controlled weight -- it is never enforced as a hard constraint, so
    feedback loops remain representable (SI Section 1.1).

    ``scale`` divides ``A o A`` before the exponential (``h = Tr exp(A o A / s) - d``,
    still zero iff DAG). With dense ``Pi`` (e.g. the ``Pi = 0.5`` initialisation of
    Algorithm 1) the unscaled ``h`` grows like ``exp(d/4)``; ``scale=None`` uses
    ``s = 1`` and should only be used for small ``d``.
    """
    from scipy.linalg import expm

    s = 1.0 if scale is None else float(scale)
    E = expm(Pi * Pi / s)
    h = float(np.trace(E)) - Pi.shape[0]
    grad = E.T * 2.0 * Pi / s
    return h, grad
