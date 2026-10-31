"""Variational Bayesian Graph Neural Network (VB-GNN), Algorithm 1 of the SI.

Pure NumPy implementation with analytic gradients (verified against finite
differences in ``tests/test_gradients.py``).

Generative model (SI Section 1.1)
---------------------------------
* Latent graph ``Z in {0,1}^{d x d}`` with variational posterior
  ``q_phi(Z) = prod_ij Bernoulli(pi_ij)``; ``Pi = [pi_ij]`` is optimised directly
  and projected onto ``[eps_Pi, 1 - eps_Pi]`` after every step (Algorithm 1, l.12-13).
* Relaxed samples ``Z_hat ~ BinaryConcrete(Pi, tau)`` (Gumbel-Softmax, Eq. 9).
* Non-linear structural equations implemented by a message-passing GNN decoder::

      a_k   = sum_j Z_hat[j, k] * x_j * W1[k, j, :] + b1[k]      (messages from parents)
      mu_k  = w2[k] . sigma(a_k) + b2[k]  (+ nuisance head  c . U[:, k])
      x_k   = mu_k + eps_k,   eps_k ~ N(0, exp(s_k))

  i.e. ``H^{(1)} = sigma(Z_hat^T H^{(0)} W^{(0)})`` with node-specific weights, which
  is the per-node additive-noise SEM ``x_j = f_theta_j(x_Pa_j) + eps_j`` of the SI.
  Self-loops are excluded; directed cycles are allowed (pseudo-likelihood, Eq. 3).

Objective (Algorithm 1, l.11)::

    L_total = -ELBO + L_struct + w_acyc * h(Pi) + w_adv * L_adv
    -ELBO   = E_q[-ln p(X | Z_hat)]  +  kl_weight * KL(q || p)

``KL`` uses the Gaussian surrogate of Assumption (A2) (default) or the exact
Bernoulli KL on the clipped domain. ``h`` is a *soft* acyclicity prior
``p(Z) propto exp(-gamma h(Z))`` -- by default the expected number of 2-cycles,
optionally the (scaled) NOTEARS characteristic; weight 0 disables it and it is
never a hard constraint, so feedback loops remain representable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .adversarial import Discriminator
from .optim import Adam
from .regularizers import (acyclicity, egs_struct_loss, offdiag_mask,
                           shannon_struct_loss, structural_diversity, two_cycle_penalty)
from .relaxation import concrete_sample, logit, sample_logistic_noise

LOG2PI = float(np.log(2.0 * np.pi))


@dataclass
class VBGNNConfig:
    hidden: int = 16                 # hidden units per node MLP
    # structural regularizer ----------------------------------------------------
    regularizer: str = "egs"         # "egs" | "shannon" | "none"
    lam: float = 0.03                # lambda (overall structural weight)
    beta: float = 1.0                # beta (dependency multiplier)
    normalize_struct: bool = True    # D = 1 - ||Pi||_F^2 / m, m = d(d-1) (see regularizers.py)
    shrinkage: float = 0.1           # s in Sigma = (1-s) Sigma_hat + s I
    # prior / KL ----------------------------------------------------------------
    kl: str = "gaussian"             # "gaussian" (Assumption A2) | "bernoulli"
    sigma0: float = 0.3              # std of the Gaussian surrogate prior
    prior_in: float = 0.75           # prior edge probability for curated (KEGG) edges
    prior_out: float = 0.05          # prior edge probability for all other edges
    kl_weight: float | None = None   # None -> 1 / n_train (per-sample ELBO scaling)
    acyc_weight: float = 30.0        # gamma: weight of the soft acyclicity prior (0 disables)
    acyc_prior: str = "notears"      # "notears": Tr exp(Pi o Pi / d) - d | "two_cycle": sum_{i<j} pi_ij pi_ji
    # relaxation ------------------------------------------------------------------
    tau: float = 1.0                 # Concrete temperature, must be in (0, 1]
    n_mc: int = 1                    # K Monte-Carlo graph samples per step
    eps_pi: float = 1e-3             # clipping margin eps_Pi
    # adversarial surrogate-variable module -------------------------------------
    adv_weight: float = 1.0
    adv_hidden: int = 32
    adv_eta: float = 0.05
    adv_warmup: int = 200
    n_critic: int = 1
    # optimisation --------------------------------------------------------------
    lr: float = 3e-3                 # alpha for decoder / nuisance head / discriminator
    lr_pi: float = 3e-3              # alpha for the edge probabilities
    iters: int = 4000
    batch_size: int = 256
    seed: int = 0
    log_every: int = 100
    extra: dict = field(default_factory=dict)


class VBGNN:
    """VB-GNN causal discovery model."""

    def __init__(self, config: VBGNNConfig | None = None, **kwargs):
        self.config = config or VBGNNConfig(**kwargs)
        if not 0.0 < self.config.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1] (SI Proposition 1).")
        self.params: dict[str, np.ndarray] = {}
        self.history: list[dict] = []
        self.feature_names: list[str] | None = None
        self.Sigma: np.ndarray | None = None
        self.A_prior: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    # ------------------------------------------------------------------ setup --
    def _init(self, d: int, r: int, A_prior: np.ndarray, rng: np.random.Generator) -> None:
        h = self.config.hidden
        self.mask = offdiag_mask(d)
        # Algorithm 1, line 1: pi_ij <- Sigmoid([A_prior]_ij)
        Pi = 1.0 / (1.0 + np.exp(-A_prior.astype(float)))
        Pi = np.clip(Pi, self.config.eps_pi, 1 - self.config.eps_pi) * self.mask
        scale = 1.0 / np.sqrt(max(1.0, 0.5 * d))
        self.params = {
            "Pi": Pi,
            "W1": rng.normal(0.0, scale, (d, d, h)),
            "b1": np.zeros((d, h)),
            "w2": rng.normal(0.0, 1.0 / np.sqrt(h), (d, h)),
            "b2": np.zeros(d),
            "logvar": np.zeros(d),
            "U": np.zeros((r, d)),
        }
        self.mu0 = np.where(A_prior > 0, self.config.prior_in, self.config.prior_out) * self.mask

    # ---------------------------------------------------------------- decoder --
    def _forward(self, X: np.ndarray, Z: np.ndarray):
        p = self.params
        d, _, h = p["W1"].shape
        V = Z.T[:, :, None] * p["W1"]                         # V[k, j, :] = Z[j, k] W1[k, j, :]
        A = (X @ V.transpose(1, 0, 2).reshape(d, d * h)).reshape(len(X), d, h) + p["b1"]
        H = np.tanh(A)
        mu_c = np.einsum("nkh,kh->nk", H, p["w2"]) + p["b2"]
        return mu_c, H

    def _backward(self, X: np.ndarray, Z: np.ndarray, H: np.ndarray, dmu: np.ndarray):
        p = self.params
        n, d, h = H.shape
        g = {"b2": dmu.sum(0), "w2": np.einsum("nk,nkh->kh", dmu, H)}
        dA = dmu[:, :, None] * p["w2"][None] * (1.0 - H * H)
        g["b1"] = dA.sum(0)
        G = (X.T @ dA.reshape(n, d * h)).reshape(d, d, h)     # G[j, k, :]
        g["W1"] = G.transpose(1, 0, 2) * Z.T[:, :, None]
        dZ = np.sum(G * p["W1"].transpose(1, 0, 2), axis=2)  # dZ[j, k]
        return g, dZ

    def _nll(self, X, mu):
        lv = self.params["logvar"]
        R = X - mu
        iv = np.exp(-lv)
        nll = 0.5 * float(np.mean(np.sum(R * R * iv + lv + LOG2PI, axis=1)))
        dmu = -(R * iv) / len(X)
        dlv = 0.5 * np.mean(1.0 - R * R * iv, axis=0)
        return nll, dmu, dlv

    # ---------------------------------------------------------- loss pieces --
    def _kl(self, Pi):
        c = self.config
        if c.kl == "gaussian":
            diff = (Pi - self.mu0) * self.mask
            return float(np.sum(diff ** 2)) / (2 * c.sigma0 ** 2), diff / c.sigma0 ** 2
        P = np.clip(Pi, c.eps_pi, 1 - c.eps_pi)
        rho = np.clip(self.mu0, c.eps_pi, 1 - c.eps_pi)
        kl = P * np.log(P / rho) + (1 - P) * np.log((1 - P) / (1 - rho))
        return float(np.sum(kl * self.mask)), (logit(P) - logit(rho)) * self.mask

    def _struct(self, Pi):
        c = self.config
        if c.regularizer == "egs":
            return egs_struct_loss(Pi, self.Sigma, c.lam, c.beta, c.normalize_struct)
        if c.regularizer == "shannon":
            loss, g = shannon_struct_loss(Pi, c.lam)
            return loss, g * self.mask
        return 0.0, np.zeros_like(Pi)

    # -------------------------------------------------------------------- fit --
    def fit(self, X: np.ndarray, A_prior: np.ndarray | None = None,
            C: np.ndarray | None = None, Sigma: np.ndarray | None = None,
            feature_names: list[str] | None = None, X_val: np.ndarray | None = None,
            verbose: bool = False, init: dict | None = None) -> "VBGNN":
        """Run Algorithm 1.

        Parameters
        ----------
        X : (n, d) preprocessed data (features standardised).
        A_prior : (d, d) binary prior adjacency (e.g. KEGG); zeros if None.
        C : (n, r) nuisance variables (measured covariates and/or surrogate
            variables). If given, a linear nuisance head absorbs their effect in the
            likelihood and the adversary enforces ``Z_causal  _|_  Z_nuisance``.
        Sigma : (d, d) feature covariance; shrinkage estimate from ``X`` if None.
        init : optional parameter dict (e.g. ``other_model.params``) used as a warm
            start; entries whose shapes match replace the random initialisation.
            Used to fit cell-type-specific models from a pooled fit.
        """
        c = self.config
        rng = np.random.default_rng(c.seed)
        X = np.asarray(X, float)
        n, d = X.shape
        A_prior = np.zeros((d, d)) if A_prior is None else np.asarray(A_prior, float)
        C = np.zeros((n, 0)) if C is None else np.asarray(C, float).reshape(n, -1)
        r = C.shape[1]
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(d)]
        self.A_prior = A_prior
        if Sigma is None:
            Sigma = shrinkage_covariance(X, c.shrinkage)
        self.Sigma = Sigma
        self._init(d, r, A_prior, rng)
        if init:
            for k, v in init.items():
                if k in self.params and np.shape(v) == self.params[k].shape:
                    self.params[k] = np.array(v, float, copy=True)
        kl_w = c.kl_weight if c.kl_weight is not None else 1.0 / n

        model_opt = Adam(self.params, lr=c.lr, lr_overrides={"Pi": c.lr_pi})
        use_adv = r > 0 and c.adv_weight > 0
        if use_adv:
            self.disc = Discriminator(d + r, c.adv_hidden, c.adv_eta, rng)
            disc_opt = Adam(self.disc.params, lr=c.lr, maximize=True)

        self.history = []
        bs = min(c.batch_size, n)
        for it in range(1, c.iters + 1):
            idx = rng.choice(n, bs, replace=False) if bs < n else np.arange(n)
            Xb, Cb = X[idx], C[idx]
            Pi = self.params["Pi"]
            grads = {k: np.zeros_like(v) for k, v in self.params.items()}
            nll_tot, adv_val = 0.0, 0.0
            for k in range(c.n_mc):
                noise = sample_logistic_noise(Pi.shape, rng)
                Z, dZdPi = concrete_sample(np.clip(Pi, c.eps_pi, 1 - c.eps_pi), c.tau, noise)
                Z = Z * self.mask
                mu_c, H = self._forward(Xb, Z)
                mu = mu_c + Cb @ self.params["U"]
                nll, dmu, dlv = self._nll(Xb, mu)
                dmu_c = dmu.copy()
                if use_adv and it > c.adv_warmup and k == 0:
                    perm = rng.permutation(bs)
                    for _ in range(c.n_critic):
                        adv_val, g_psi, dzc = self.disc.objective(mu_c, Cb, perm)
                        disc_opt.step(g_psi)
                        self.disc.spectral_normalize()
                    adv_val, _, dzc = self.disc.objective(mu_c, Cb, perm)
                    dmu_c += c.adv_weight * dzc
                g, dZ = self._backward(Xb, Z, H, dmu_c)
                for key, val in g.items():
                    grads[key] += val / c.n_mc
                grads["Pi"] += (dZ * dZdPi * self.mask) / c.n_mc
                grads["logvar"] += dlv / c.n_mc
                grads["U"] += (Cb.T @ dmu) / c.n_mc
                nll_tot += nll / c.n_mc

            kl, g_kl = self._kl(Pi)
            st, g_st = self._struct(Pi)
            grads["Pi"] += kl_w * g_kl + g_st * self.mask
            acyc = 0.0
            if c.acyc_weight > 0:
                acyc, g_h = (two_cycle_penalty(Pi) if c.acyc_prior == "two_cycle"
                             else acyclicity(Pi, scale=Pi.shape[0]))
                grads["Pi"] += c.acyc_weight * g_h * self.mask
            model_opt.step(grads)
            self.params["Pi"] = np.clip(self.params["Pi"], c.eps_pi, 1 - c.eps_pi) * self.mask
            model_opt.params = self.params

            if it % c.log_every == 0 or it == c.iters:
                rec = {"iter": it, "nll": nll_tot, "kl": kl, "struct": st, "acyc": acyc,
                       "adv": adv_val, "D": structural_diversity(Pi, c.normalize_struct),
                       "total": nll_tot + kl_w * kl + st + c.acyc_weight * acyc + c.adv_weight * adv_val}
                if X_val is not None:
                    rec["val_nll"] = self.score_nll(X_val)
                self.history.append(rec)
                if verbose:
                    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in rec.items()))
        return self

    # --------------------------------------------------------------- results --
    @property
    def edge_probs(self) -> np.ndarray:
        """Posterior edge probabilities ``pi_hat`` (``[i, j]`` = P(i -> j | X))."""
        return self.params["Pi"] * self.mask

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None, C: np.ndarray | None = None) -> np.ndarray:
        """Structural mean ``mu(X)`` using the expected graph (``Z = Pi``) by default."""
        Z = self.edge_probs if Z is None else Z
        mu, _ = self._forward(np.asarray(X, float), Z * self.mask)
        if C is not None and self.params["U"].size:
            mu = mu + np.asarray(C, float) @ self.params["U"]
        return mu

    def score_nll(self, X: np.ndarray, Z: np.ndarray | None = None) -> float:
        mu = self.predict(X, Z)
        return self._nll(np.asarray(X, float), mu)[0]

    def jacobian(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Per-sample Jacobian ``J[n, k, j] = d mu_k / d x_j`` of the GNN map."""
        Z = (self.edge_probs if Z is None else Z) * self.mask
        _, H = self._forward(np.asarray(X, float), Z)
        G = self.params["w2"][None] * (1.0 - H * H)          # (n, k, h)
        WZ = self.params["W1"] * Z.T[:, :, None]              # (k, j, h)
        return np.einsum("nkh,kjh->nkj", G, WZ)

    def signed_effects(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Average direct effect ``E_n[d mu_j / d x_i]`` arranged as ``[i, j]`` (source, target)."""
        return self.jacobian(X, Z).mean(0).T

    # ------------------------------------------------------------ persistence --
    def save(self, path: str) -> None:
        meta = {"config": asdict(self.config), "feature_names": self.feature_names,
                "history": self.history}
        arrays = {f"param_{k}": v for k, v in self.params.items()}
        arrays.update(Sigma=self.Sigma, A_prior=self.A_prior, mu0=self.mu0)
        if self.x_mean is not None:
            arrays.update(x_mean=self.x_mean, x_std=self.x_std)
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str) -> "VBGNN":
        f = np.load(path, allow_pickle=False)
        meta = json.loads(str(f["meta"]))
        model = cls(VBGNNConfig(**meta["config"]))
        model.params = {k[6:]: f[k] for k in f.files if k.startswith("param_")}
        model.feature_names = meta["feature_names"]
        model.history = meta["history"]
        model.Sigma, model.A_prior, model.mu0 = f["Sigma"], f["A_prior"], f["mu0"]
        model.mask = offdiag_mask(model.params["Pi"].shape[0])
        if "x_mean" in f.files:
            model.x_mean, model.x_std = f["x_mean"], f["x_std"]
        return model


def shrinkage_covariance(X: np.ndarray, s: float = 0.1) -> np.ndarray:
    """``Sigma = (1 - s) Sigma_hat + s I`` (SI Section 1.2), always in S^d_{++}."""
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / max(1, len(X) - 1)
    return (1.0 - s) * S + s * np.eye(X.shape[1])
