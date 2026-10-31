# OmicsDepth VB-GNN

**Dependency-aware multi-omic causal discovery with Variational Bayesian Graph Neural Networks (VB-GNN)
and Entropy-based Gini–Simpson (EGS) structural regularization.**

Code accompanying

> S. Jeon, H. Chun, E. Yun and J. Kang. *A Mitochondrial–Epigenetic Feedback Loop Linked to Synaptic
> Gene Repression in Schizophrenia Uncovered by Multi-Omic Causal Discovery* (2026).
> Supplementary Information: <https://zenodo.org/records/23794101>

VB-GNN learns a directed graph over the features of several molecular layers measured on the same samples
(genotype, histone ChIP-seq, snRNA-seq pseudobulk, LC-MS metabolomics). It

1. keeps **feedback loops** by treating acyclicity as a soft prior rather than a hard constraint;
2. replaces Shannon-entropy structural regularization with the **EGS** term `L_struct = λβ D Tr(ΠᵀΣΠ)`, whose
   gradient and Hessian are uniformly bounded on the whole edge-probability hypercube (SI Proposition 1, Theorem 1);
3. removes measured technical and demographic confounding (PMI, age, sex, batch) with an **adversarial
   surrogate-variable module**;
4. selects edges with a **posterior-FDR rule** and supports cell-type causal flux, equilibrium do-interventions
   on cyclic graphs, GSEA and Causal Match Score drug ranking.

The core is pure NumPy/SciPy with analytic gradients, each checked against finite differences.

## Installation

```bash
git clone https://github.com/JoonsungKang/OmicsDepth-VB-GNN-SCHIZOPHRENIA.git
cd OmicsDepth-VB-GNN-SCHIZOPHRENIA
pip install -r requirements.txt            # numpy, scipy, pandas, matplotlib, pyyaml, pytest
pytest -q                                  # 27 tests: gradients, SI theorems, FDR, pipeline smoke test
# optional, for the deep-learning baselines of Table 1 / Figure 2:
pip install -r requirements-baselines.txt  # torch, gcastle (DAG-GNN, GraN-DAG), causal-learn
```

Python ≥ 3.9. No GPU is needed.

## Quick start

```python
import numpy as np
from vbgnn import VBGNN, VBGNNConfig, posterior_fdr_select, structure_metrics
from vbgnn.simulation import make_graph, simulate_anm

rng = np.random.default_rng(0)
B = make_graph("ER-2", 15, rng)                       # B[i, j] = 1  <=>  i -> j
X = simulate_anm(B, n=1000, rng=rng, rho=0.3)         # non-linear ANM, correlated noise

model = VBGNN(VBGNNConfig(iters=3000)).fit(X)          # optional: A_prior=..., C=nuisance covariates
selected, efdp = posterior_fdr_select(model.edge_probs, alpha=0.05)
print(structure_metrics(B, selected))
```

## Reproducing the analyses

| Paper | Command | Output |
|---|---|---|
| Table 1 (SHD / FDR, ER-2, ER-4, SF-4) | `python scripts/run_simulation.py --config configs/simulation.yaml --jobs 16` | `results/simulation/table1.md` |
| Figure 2 (FDR vs ρ, SHD vs σ²) | `python scripts/run_ablation.py --config configs/ablation.yaml --jobs 16` | `results/ablation/figure2.png` |
| Sections 4–5 (SCZ pipeline) | `python scripts/run_scz_pipeline.py --config configs/scz_demo.yaml` | `results/scz_demo/` |
| Theory (Prop. 1, Thm. 1–2) | `pytest tests/test_theory.py` or `notebooks/02_theory_checks.ipynb` | |

`configs/*_quick.yaml` run the same benchmarks at `d = 20` with 3 trials in minutes. The full configurations
(`d = 100`, `n = 1000`, 50 trials) take several CPU-hours; VB-GNN takes about one minute per fit at `d = 100`.
DCDI is run with its official code: export the simulated data with `--export-data DIR`, run DCDI, and point
`DCDI_RESULTS` at the learned adjacency matrices (see `baselines/external.py`). Methods whose dependencies are
missing are reported and skipped.

### Bundled results (quick configurations, this repository's code)

These were produced by the commands above with the `*_quick.yaml` configurations, on a CPU-only machine
**without** PyTorch, so DAG-GNN, GraN-DAG and DCDI were not run. They are smaller-scale runs (`d = 20`, 3 trials)
than the paper's Table 1 and Figure 2, so the numbers are not comparable one-to-one.

**Table 1, quick** (`results/simulation_quick/table1.md`; mean ± SD over 3 trials, `d = 20`, `n = 1000`):

| Method | ER-2 SHD | ER-2 FDR | ER-4 SHD | ER-4 FDR | SF-4 SHD | SF-4 FDR |
|---|---|---|---|---|---|---|
| PC | 34.3 ± 9.8 | 0.34 ± 0.11 | 82.0 ± 10.1 | 0.31 ± 0.27 | 69.0 ± 1.7 | 0.65 ± 0.22 |
| NOTEARS | 33.7 ± 6.7 | 0.42 ± 0.24 | 81.0 ± 12.5 | 0.37 ± 0.23 | 63.3 ± 0.6 | 0.30 ± 0.11 |
| VB-GNN | 22.0 ± 9.8 | 0.17 ± 0.10 | 67.7 ± 11.2 | 0.18 ± 0.15 | 39.7 ± 7.8 | 0.19 ± 0.06 |

**Figure 2, quick** (`results/ablation_quick/figure2.png`):
*(a) FDR vs noise correlation ρ (ER-2, mean over 3 trials).*

| ρ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| VB-GNN (EGS) | 0.17 | 0.17 | 0.19 | 0.20 | 0.35 |
| VB-GNN with Shannon regularizer | 0.43 | 0.44 | 0.53 | 0.61 | 0.67 |
| NOTEARS | 0.28 | 0.44 | 0.52 | 0.67 | 0.75 |

EGS keeps the FDR lower than the Shannon variant of the same model and than NOTEARS at every ρ. Its FDR is
flat up to ρ = 0.6 but rises to 0.35 at ρ = 0.8; it does not stay at or below 0.05 in these runs.

*(b) SHD vs noise variance σ².* VB-GNN has the lowest SHD at σ² = 0.5 and 1.0 on both ER-2 and SF-4. At
σ² = 0.1, NOTEARS is better (ER-2 29.3 vs 33.3; SF-4 64.7 vs 74.7). At σ² ≥ 1.5 the Shannon variant is similar
or better (SF-4, σ² = 2: 54.3 vs 64.0). The curve is U-shaped rather than monotone. Full values are in
`results/ablation_quick/sigma2_sweep.csv`.

**SCZ demo** (`results/scz_demo/report.md`; synthetic cohort with a planted loop, see below):

- 17 edges selected at posterior FDR 0.05. Against the planted graph: SHD 23, FDR 0.29, TPR 0.35
  (5 of the 17 are reversed true edges).
- Directional AUC on the 25 held-out reference relations: **VB-GNN 0.86**, NOTEARS 0.82, absolute
  correlation 0.79.
- Succinate → GRIN2A/DISC1 causal flux by cell type: PV+ 1.03, L2/3 1.00, astrocytes 0.80, microglia 0.44,
  oligodendrocytes 0.33. This matches the planted ordering.
- `do(Succinate = control level)` recovers 77–82 % of the GRIN2A/DISC1 case–control gap in L2/3 neurons. In PV+
  neurons the predicted recovery is 100–114 %, so the PV+-specific model overshoots.
- Causal Match Score (PV+): Metformin −0.70, Dimethyl Fumarate −0.56, dimethyl-αKG −0.37. Vorinostat (−0.16)
  ranks below two random decoys.
- GSEA of the causal-regulator score ranks Mito-Metabolic Stress first (NES 1.17), but no set is significant with
  only 25 features (all p ≥ 0.25).

## The schizophrenia pipeline

`scripts/run_scz_pipeline.py` (implemented in `vbgnn/pipeline.py`) runs ten steps:

1. per-layer QC, log2, normalisation; layers matched on `sample_id`, centred within cell type, standardised;
2. nuisance variables (measured covariates, optional surrogate variables) → adversarial module (or regressed out);
3. donor-level split into training and held-out partitions;
4. VB-GNN fit with the curated prior `A_prior` (checkpoint saved);
5. posterior-FDR edge selection with signed direct effects;
6. directional ROC against reference relations that were withheld from the prior, compared with other methods;
7. preranked GSEA of the causal-regulator score;
8. cell-type causal flux, computed with cell-type-specific refits warm-started from the pooled model;
9. counterfactual `do(Succinate = control level)` solved at the equilibrium of the cyclic SEM;
10. Causal Match Scores of compound signatures against the cell-type disease causal-flux vector.

### Data

The PsychENCODE / CommonMind DLPFC profiles analysed in the paper are **controlled-access** (PsychENCODE
Knowledge Portal, Synapse syn20821315 and syn21415492; CommonMind syn2755792) and are **not** included.
`docs/DATA.md` describes the input format, and `configs/scz_psychencode.yaml` is the template for running the
pipeline on them once access is granted.

The repository ships a **synthetic** cohort in `data/demo/` (600 donors × 5 cell types, 25 features, four
layers) generated by `scripts/make_demo_data.py`. It has a planted mitochondrial–epigenetic–synaptic feedback
loop that is stronger in PV+ and L2/3 neurons, plus PMI, batch, age and sex confounding. It exercises every
step and checks that a known loop can be recovered. **Results on it are not evidence about human biology.**
`checkpoints/vbgnn_scz_demo.npz` is the model fitted to it.

## Repository layout

```
vbgnn/            core package
  model.py          VB-GNN (SI Algorithm 1): Concrete relaxation, GNN decoder, ELBO, priors, Adam, projection
  regularizers.py   EGS structural loss, gradient/Hessian and their bounds; Shannon; acyclicity priors
  adversarial.py    spectrally normalised discriminator with output margin; surrogate variables
  fdr.py            posterior-FDR rule; Benjamini–Hochberg / Benjamini–Yekutieli
  causal.py         causal flux, equilibrium do-interventions, cell-type-specific refits
  translational.py  Causal Match Score, preranked GSEA
  preprocessing.py  per-layer QC / log2 / quantile normalisation, pseudobulk, covariates
  simulation.py     ER / SF DAGs, non-linear ANM (GP, MLP, periodic), correlated noise
  metrics.py        SHD, FDR, TPR, AUROC, directional AUROC
  benchmark.py      shared runner for the benchmarks
  pipeline.py       ten-step multi-omic pipeline;  plots.py  its figures
  demo.py           synthetic SCZ cohort with a planted causal graph
baselines/        PC (Fisher-z), linear NOTEARS; adapters for DAG-GNN, GraN-DAG (gCastle) and DCDI
scripts/          run_simulation.py, run_ablation.py, run_scz_pipeline.py, make_demo_data.py, make_figures.py
configs/          YAML configurations (full and *_quick)
notebooks/        01_quickstart, 02_theory_checks, 03_scz_demo
tests/            finite-difference gradient checks, SI theorems, components, pipeline smoke test
data/demo/        synthetic demo cohort;   checkpoints/  pre-trained demo model
results/          outputs of the quick benchmarks and of the demo pipeline
docs/             METHODS.md (SI equation → code map, implementation choices), DATA.md (input formats)
```

## Implementation notes

`docs/METHODS.md` maps every SI equation to the code and lists the choices that the SI leaves open: the
scaling of `D`, the form of the soft acyclicity prior, what the adversary sees, the surrogate-variable caveat,
cell-type-specific refits and the default hyperparameters. The defaults were set on small pilot simulations;
re-tune them for new data.

## Citation

See `CITATION.cff`. Released under the MIT License (`LICENSE`).
