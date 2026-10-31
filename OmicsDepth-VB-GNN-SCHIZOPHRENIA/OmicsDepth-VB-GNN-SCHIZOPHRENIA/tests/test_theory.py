"""Numerical checks of the theoretical guarantees in the Supplementary Information.

* Proposition 1 -- |dL_struct/dpi_ij| <= M_struct on the whole hypercube, while the
  Shannon gradient diverges at the boundary.
* Theorem 1 -- (i) ||Hess L_struct||_2 <= 12 lambda beta lambda_max(Sigma) d^2;
  (ii) Hess at Pi = 0 equals 2 lambda beta (I kron Sigma) with condition number
  kappa(Sigma), and the perturbation bound for ||Pi||_F^2 = rho < 1/(6 kappa);
  (iii) Shannon curvature is unbounded.
* Theorem 2 -- D and L_struct are invariant under node relabelling.
* Eq. (6) -- the curvature of beta*D is the constant -2 beta.
"""
import numpy as np
import pytest

from vbgnn.regularizers import (egs_gradient_bound, egs_hessian_bound, egs_struct_hessian,
                                egs_struct_loss, shannon_curvature, shannon_struct_loss,
                                structural_diversity)


def rand_cov(d, rng):
    A = rng.normal(size=(d, d))
    return A @ A.T / d + 0.2 * np.eye(d)


@pytest.mark.parametrize("normalize", [False, True])
def test_prop1_gradient_bound_including_boundary(normalize):
    rng = np.random.default_rng(0)
    d, lam, beta = 5, 0.7, 1.3
    S = rand_cov(d, rng)
    M = egs_gradient_bound(S, lam, beta, d, normalize)
    worst = 0.0
    for _ in range(300):
        P = rng.uniform(size=(d, d))
        P[rng.uniform(size=(d, d)) < 0.3] = 0.0           # exact boundary values
        P[rng.uniform(size=(d, d)) < 0.2] = 1.0
        worst = max(worst, np.abs(egs_struct_loss(P, S, lam, beta, normalize)[1]).max())
    for P in (np.zeros((d, d)), np.ones((d, d))):          # corners of the hypercube
        worst = max(worst, np.abs(egs_struct_loss(P, S, lam, beta, normalize)[1]).max())
    assert worst <= M + 1e-9
    if not normalize:   # implemented bound uses |D| <= d^2 - 1, so it is within the SI constant
        lmax = np.linalg.eigvalsh(S)[-1]
        assert M <= 2 * lam * beta * lmax * (d ** 2 + d ** 2.5)


def test_prop1_shannon_gradient_diverges():
    g = [abs(shannon_struct_loss(np.full((2, 2), p), 1.0, eps=1e-300)[1][0, 0])
         for p in (1e-2, 1e-6, 1e-12, 1e-100)]
    assert all(b > a for a, b in zip(g, g[1:])) and g[-1] > 200


@pytest.mark.parametrize("normalize", [False, True])
def test_thm1_i_hessian_bound(normalize):
    rng = np.random.default_rng(1)
    d, lam, beta = 4, 0.5, 2.0
    S = rand_cov(d, rng)
    bound = egs_hessian_bound(S, lam, beta, d, normalize)
    if not normalize:
        assert bound <= 12 * lam * beta * np.linalg.eigvalsh(S)[-1] * d ** 2   # SI Theorem 1(i)
    for _ in range(100):
        P = rng.uniform(size=(d, d))
        assert np.linalg.norm(egs_struct_hessian(P, S, lam, beta, normalize), 2) <= bound + 1e-9
    assert np.linalg.norm(egs_struct_hessian(np.ones((d, d)), S, lam, beta, normalize), 2) <= bound + 1e-9


def test_thm1_ii_conditioning_at_zero_and_perturbation():
    rng = np.random.default_rng(2)
    d, lam, beta = 4, 0.8, 1.1
    S = rand_cov(d, rng)
    ev = np.linalg.eigvalsh(S)
    kappa = ev[-1] / ev[0]
    H0 = egs_struct_hessian(np.zeros((d, d)), S, lam, beta, normalize=False)
    assert np.allclose(H0, 2 * lam * beta * np.kron(np.eye(d), S))
    assert np.isclose(np.linalg.cond(H0), kappa)
    for _ in range(50):
        P = rng.uniform(size=(d, d))
        rho = 0.9 / (6 * kappa)                            # ||Pi||_F^2 = rho < 1 / (6 kappa)
        P *= np.sqrt(rho) / np.linalg.norm(P)
        H = egs_struct_hessian(P, S, lam, beta, normalize=False)
        assert np.linalg.norm(H - H0, 2) <= 12 * lam * beta * ev[-1] * rho + 1e-9
        e = np.linalg.eigvalsh((H + H.T) / 2)
        assert e[0] > 0
        assert e[-1] / e[0] <= kappa * (1 + 6 * rho) / (1 - 6 * rho * kappa) + 1e-9


def test_thm1_iii_shannon_curvature_unbounded():
    c = shannon_curvature(np.array([1e-1, 1e-4, 1e-8, 1e-12]))
    assert np.all(np.diff(c) > 0) and c[-1] > 1e11


def test_eq6_constant_curvature_of_D():
    rng = np.random.default_rng(3)
    beta, h = 1.7, 1e-4
    for _ in range(20):
        P = rng.uniform(size=(3, 3))
        E = np.zeros_like(P)
        E[1, 2] = h
        second = (beta * structural_diversity(P + E, False) - 2 * beta * structural_diversity(P, False)
                  + beta * structural_diversity(P - E, False)) / h ** 2
        assert np.isclose(second, -2 * beta, atol=1e-5)


def test_thm2_permutation_invariance():
    rng = np.random.default_rng(4)
    d = 6
    S = rand_cov(d, rng)
    P = rng.uniform(size=(d, d))
    for _ in range(10):
        Pm = np.eye(d)[rng.permutation(d)]
        Q = np.eye(d)[rng.permutation(d)]
        assert np.isclose(structural_diversity(Pm @ P @ Pm.T), structural_diversity(P))
        assert np.isclose(structural_diversity(Pm @ P @ Q.T), structural_diversity(P))
        a = egs_struct_loss(Pm @ P @ Pm.T, Pm @ S @ Pm.T, 0.3, 2.0)[0]
        assert np.isclose(a, egs_struct_loss(P, S, 0.3, 2.0)[0])
