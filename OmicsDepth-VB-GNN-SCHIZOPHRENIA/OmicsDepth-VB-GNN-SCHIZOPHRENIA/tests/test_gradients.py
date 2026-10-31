"""Finite-difference checks of every analytic gradient used by the optimiser."""
import numpy as np
import pytest

from vbgnn.adversarial import Discriminator
from vbgnn.model import VBGNN, VBGNNConfig
from vbgnn.regularizers import acyclicity, egs_struct_hessian, egs_struct_loss, shannon_struct_loss
from vbgnn.relaxation import concrete_sample


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def rand_cov(d, rng):
    A = rng.normal(size=(d, d))
    return A @ A.T / d + 0.1 * np.eye(d)


@pytest.mark.parametrize("normalize", [True, False])
def test_egs_gradient_and_hessian(normalize):
    rng = np.random.default_rng(0)
    d = 4
    S = rand_cov(d, rng)
    P = rng.uniform(0.05, 0.95, (d, d))
    _, g = egs_struct_loss(P, S, 0.7, 1.3, normalize)
    gn = num_grad(lambda: egs_struct_loss(P, S, 0.7, 1.3, normalize)[0], P)
    assert np.allclose(g, gn, atol=1e-6)
    H = egs_struct_hessian(P, S, 0.7, 1.3, normalize)
    # numerical Hessian from analytic gradients (column-major vec)
    Hn = np.zeros((d * d, d * d))
    for c in range(d * d):
        j, i = divmod(c, d)
        E = np.zeros((d, d)); E[i, j] = 1e-6
        gp = egs_struct_loss(P + E, S, 0.7, 1.3, normalize)[1]
        gm = egs_struct_loss(P - E, S, 0.7, 1.3, normalize)[1]
        Hn[:, c] = ((gp - gm) / 2e-6).reshape(-1, order="F")
    assert np.allclose(H, Hn, atol=1e-5)


def test_shannon_and_acyclicity_gradients():
    rng = np.random.default_rng(1)
    P = rng.uniform(0.1, 0.9, (3, 3))
    _, g = shannon_struct_loss(P, 0.5, normalize=False)
    assert np.allclose(g, num_grad(lambda: shannon_struct_loss(P, 0.5, normalize=False)[0], P), atol=1e-6)
    _, g = acyclicity(P)
    assert np.allclose(g, num_grad(lambda: acyclicity(P)[0], P), atol=1e-5)


def test_concrete_derivative():
    rng = np.random.default_rng(2)
    P = rng.uniform(0.05, 0.95, (3, 3))
    L = rng.logistic(size=(3, 3))
    _, dZ = concrete_sample(P, 0.5, L)
    for i in range(3):
        for j in range(3):
            E = np.zeros_like(P); E[i, j] = 1e-7
            num = (concrete_sample(P + E, 0.5, L)[0][i, j] - concrete_sample(P - E, 0.5, L)[0][i, j]) / 2e-7
            assert np.isclose(dZ[i, j], num, rtol=1e-4, atol=1e-8)


def test_decoder_backward_matches_finite_differences():
    rng = np.random.default_rng(3)
    n, d = 7, 4
    m = VBGNN(VBGNNConfig(hidden=3))
    m._init(d, 0, np.zeros((d, d)), rng)
    for k in ("b1", "b2"):
        m.params[k] = rng.normal(size=m.params[k].shape)
    X = rng.normal(size=(n, d))
    Z = rng.uniform(size=(d, d)) * (1 - np.eye(d))
    T = rng.normal(size=(n, d))

    def loss():
        mu, _ = m._forward(X, Z)
        return float(np.sum(mu * T))

    mu, H = m._forward(X, Z)
    g, dZ = m._backward(X, Z, H, T)
    for k in ("W1", "b1", "w2", "b2"):
        assert np.allclose(g[k], num_grad(loss, m.params[k]), atol=1e-6), k
    assert np.allclose(dZ * (1 - np.eye(d)), num_grad(loss, Z) * (1 - np.eye(d)), atol=1e-6)
    # Jacobian of the GNN map w.r.t. its input
    J = m.jacobian(X, Z)
    for a in range(n):
        for j in range(d):
            E = np.zeros_like(X); E[a, j] = 1e-6
            num = (m._forward(X + E, Z)[0][a] - m._forward(X - E, Z)[0][a]) / 2e-6
            assert np.allclose(J[a, :, j], num, atol=1e-6)


def test_nll_gradients():
    rng = np.random.default_rng(4)
    m = VBGNN(VBGNNConfig())
    m.params = {"logvar": rng.normal(size=3)}
    X, mu = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    _, dmu, dlv = m._nll(X, mu)
    assert np.allclose(dmu, num_grad(lambda: m._nll(X, mu)[0], mu), atol=1e-6)
    assert np.allclose(dlv, num_grad(lambda: m._nll(X, mu)[0], m.params["logvar"]), atol=1e-6)


def test_discriminator_gradients():
    rng = np.random.default_rng(5)
    D = Discriminator(5, hidden=4, eta=0.05, rng=rng)
    zc, zn = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    perm = rng.permutation(6)
    _, g, dzc = D.objective(zc, zn, perm)
    for k in D.params:
        gn = num_grad(lambda: D.objective(zc, zn, perm)[0], D.params[k])
        assert np.allclose(g[k], gn, atol=1e-6), k
    assert np.allclose(dzc, num_grad(lambda: D.objective(zc, zn, perm)[0], zc), atol=1e-6)
