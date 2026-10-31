# From the Supplementary Information to the code

## Equation / algorithm map

| SI | Object | Code |
|---|---|---|
| Eq. (1) | ELBO `E_q[ln p(X\|Z)] − KL(q‖p)` | `VBGNN.fit` (`vbgnn/model.py`): Monte-Carlo NLL + `kl_weight · KL` |
| Eq. (2), (A2) | KL to the biological prior; Gaussian surrogate `Σ(π−μ0)²/(2σ0²)` or Bernoulli KL | `VBGNN._kl` (`kl="gaussian"` / `"bernoulli"`) |
| Eq. (3) | per-node Gaussian likelihood of the non-linear SEM `x_j = f_θj(x_Pa_j) + ε_j` | `VBGNN._forward`, `_nll` (node-wise MLP on relaxed parents, learned `σ_j²`) |
| Section 1.1 | acyclicity as a probabilistic prior, not a hard constraint | `acyc_prior` / `acyc_weight` (`regularizers.acyclicity`, `two_cycle_penalty`) |
| Eq. (4)–(5), (A3) | adversarial surrogate-variable module, `D_ψ ∈ [η, 1−η]`, spectral normalisation, `T` by row shuffling | `vbgnn/adversarial.py` (`Discriminator.objective`) |
| Section 1.2 | shrinkage covariance `Σ = (1−s)Σ̂ + sI` | `model.shrinkage_covariance` |
| Section 1.2 | structural diversity `D = 1 − ‖Π‖²_F` | `regularizers.structural_diversity` |
| Eq. (10)–(12) | Shannon regularizer (baseline / ablation) | `regularizers.shannon_struct_loss` (`regularizer="shannon"`) |
| Eq. (15), (19) | `L_struct = λβ D Tr(ΠᵀΣΠ)` and its gradient | `regularizers.egs_struct_loss` |
| Eq. (32) | full Hessian `H_EGS` | `regularizers.egs_struct_hessian` |
| Prop. 1, Eq. (22) | gradient bound `M_struct` | `regularizers.egs_gradient_bound`; checked in `tests/test_theory.py` |
| Thm. 1 (i)–(iii) | Hessian bound `12λβλmax d²`, `κ(H(0)) = κ(Σ)`, perturbation bound, Shannon curvature | `egs_hessian_bound`, `shannon_curvature`; `tests/test_theory.py` |
| Thm. 2 | permutation invariance of `D` and `L_struct` | `tests/test_theory.py::test_thm2_permutation_invariance` |
| Eq. (9) | binary Concrete relaxation `Ẑ = Sigmoid((logit π + L)/τ)`, `τ ∈ (0,1]` | `vbgnn/relaxation.py` |
| Alg. 1 l.1 | `π_ij ← Sigmoid([A_prior]_ij)` | `VBGNN._init` |
| Alg. 1 l.12–13 | Adam + projection onto `[ε_Π, 1−ε_Π]` | `vbgnn/optim.py`, `VBGNN.fit` |
| Alg. 1 l.14 | discriminator ascent | `Adam(..., maximize=True)` |
| Alg. 1 l.15, Section 1.2 | posterior-FDR rule `k* = max{k : k⁻¹Σ ℓ_(l) ≤ α}`; BH / BY for p-values | `fdr.posterior_fdr_select`, `fdr.benjamini_hochberg` |
| Alg. 1 l.16 | causal flux `[F_g]_j = ‖∂GNN(X_g)/∂x_{g,j}‖_F` | `causal.causal_flux` (analytic Jacobian `VBGNN.jacobian`) |
| Section 2.3 | per-layer preprocessing, covariates regressed out or passed as `Z_nuisance` | `vbgnn/preprocessing.py`, `pipeline.prepare_data` |
| Section 2.4 / main text 5.7 | do-interventions at the equilibrium of the cyclic SEM | `causal.counterfactual_do` (damped fixed point, abduction of residuals) |
| main text 5.7 | Causal Match Score | `translational.causal_match_score`, `disease_flux_vector` |
| main text 4.2 | preranked GSEA | `translational.preranked_gsea` |

All analytic gradients (decoder, likelihood, Concrete relaxation, EGS, Shannon, acyclicity,
discriminator) are verified against finite differences in `tests/test_gradients.py`.

## Implementation choices not fixed by the SI

These choices were needed to turn the SI into working code. Each is a configuration option.

1. **Scaling of `D`.** On `[0,1]^{d×d}` the literal `D = 1 − ‖Π‖²_F` ranges over `[1 − d², 1]`, so `L_struct`
   can be negative. With `normalize_struct=True` (default) the Frobenius term is divided by `m = d(d−1)`, which
   keeps `D ∈ [0,1]` and `L_struct ≥ 0`. Constant curvature, bounded Hessian and permutation invariance all
   still hold, with rescaled constants (`egs_hessian_bound`). `Tr(ΠᵀΣΠ)` is left unscaled so that the per-edge
   gradient `2λβD[ΣΠ]_ij` does not vanish as `d` grows. `normalize_struct=False` gives the literal SI formula.
2. **Soft acyclicity prior.** The SI treats acyclicity as a prior `p(Z)` without fixing its form. The default is
   `p(Z) ∝ exp(−γ h(Π))` with `h = Tr exp(Π∘Π/d) − d` (`acyc_prior="notears"`, `γ = acyc_weight = 30`). The `1/d`
   inside the exponential keeps `h` finite at the `π = 0.5` initialisation of Algorithm 1; the unscaled NOTEARS
   form grows like `e^{d/4}`. Because it is a penalty, not a constraint, cycles with strong support survive.
   `acyc_prior="two_cycle"` penalises only `i ↔ j` pairs and leaves every longer loop free. Without any
   acyclicity prior, the cyclic pseudo-likelihood tends to keep both directions of each true edge.
3. **What the adversary sees.** `Z_causal` is represented per sample by the structural predictions `μ_c(x)` of
   the GNN decoder and `Z_nuisance` by the measured covariates (plus optional surrogate variables). A linear
   nuisance head absorbs covariate effects in the likelihood, and the discriminator penalises any remaining
   dependence between `μ_c` and `Z_nuisance`.
4. **Surrogate variables.** `n_surrogate_variables > 0` adds the leading principal components of the data after
   removing diagnosis and measured covariates. When the biological programme of interest is itself a dominant
   axis of variation, those components *are* that programme, and treating them as nuisance removes the signal.
   On the demo cohort, two such components drop the directional AUC from 0.86 to 0.68. The default is 0.
5. **Cell-type-specific parameter spaces.** A single pooled model shares its structural functions across cell
   types, so differences in its Jacobian only reflect where each cell type sits on a shared non-linearity. For
   the cell-type flux, counterfactual and Causal Match steps, the pipeline therefore refits one model per cell
   type, warm-started from the pooled fit (`causal.fit_group_models`). With the pooled model alone, the demo's
   flux ranking was inverted relative to the planted cell-type weights.
6. **Edge selection.** Edges are selected with the posterior-FDR rule at `α = 0.05` on the converged `π̂`. This
   rule controls the *posterior expected* FDP under the fitted variational posterior; it is not a frequentist
   guarantee. The FDR measured against a known truth can be higher, especially under strongly correlated noise
   (see the README).
7. **Default hyperparameters.** The defaults (`λ = 0.03`, `β = 1`, `γ = 30`, `τ = 1`,
   `lr_pi = 3·10⁻³`, 4000 iterations) came from small pilot runs on ER-2 graphs with `d = 20`. They are not
   tuned per data set; re-tune them for new settings, for example by held-out NLL.
