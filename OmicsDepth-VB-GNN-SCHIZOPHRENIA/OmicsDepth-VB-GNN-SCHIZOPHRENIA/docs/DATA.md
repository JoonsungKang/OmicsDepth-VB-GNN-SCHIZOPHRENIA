# Input data formats

The pipeline (`scripts/run_scz_pipeline.py`, `vbgnn/pipeline.py`) reads plain CSV/JSON files
listed in a YAML config. `configs/scz_demo.yaml` points at the bundled synthetic cohort in
`data/demo/`; `configs/scz_psychencode.yaml` is the template for the controlled-access data.

## Samples table — `samples.csv`

One row per profiled sample. For cell-type-resolved analyses a sample is a
**donor × cell type pseudobulk profile**.

| column | meaning |
|---|---|
| `sample_id` | unique key shared by every layer file |
| `donor_id` | donor; the train / held-out split is made by donor |
| `cell_type` | group label `g` for the causal flux `F_g` (set `group_col`) |
| `diagnosis` | case/control label (`case_label`, e.g. `SCZ`) |
| covariates | e.g. `pmi`, `age`, `sex`, `batch` — listed under `covariates`; categorical columns are one-hot encoded |

## Layer matrices — one CSV per molecular layer

First column `sample_id`, then one column per feature, raw (unlogged) values unless `log2: false`.
Feature names must be unique across layers. Each layer is processed separately: QC filter
(`max_missing`, zero variance), optional `log2(x + 1)`, median imputation, optional within-layer
quantile normalisation; layers are then joined on `sample_id`, centred within cell type
(`center_by_group`) and standardised.

* **genotype** — SNP dosages (0/1/2) or polygenic scores, used as instrument-like anchors (`log2: false`).
* **epigenomics** — ChIP-seq signal (H3K27ac, H3K9me3) summarised per regulatory element, one column per mark × element (e.g. `H3K27ac_GRIN2A`).
* **transcriptomics** — snRNA-seq pseudobulk counts (aggregate cells with `vbgnn.preprocessing.pseudobulk`), or bulk / sorted RNA-seq.
* **metabolomics** — LC-MS abundances (e.g. succinate, α-KG, lactate).

## Prior and evaluation edges

* `prior_edges.csv` — `source,target`: curated directed relations (e.g. KEGG) mapped onto feature
  names. Becomes `A_prior` (SI Section 1.1): Algorithm 1 initialises `π_ij = Sigmoid(A_prior)` and the KL
  term pulls curated edges towards `prior_in` and all others towards `prior_out`.
* `reference_edges.csv` — `source,target`: experimentally supported directional relations used **only** for
  the directional ROC. They must not appear in `prior_edges.csv`.
* `true_edges.csv` — planted ground truth; only meaningful for simulated data.

## Gene sets and compound signatures

* `gene_sets.json` — `{"set name": [feature, ...]}` for preranked GSEA of the causal-regulator score.
* `compound_signatures.csv` — `compound,mechanism,<feature columns>`: the change in each feature induced
  by a compound (e.g. LINCS L1000 consensus signatures, on the log2 scale of the layers). Features that are
  absent are ignored. Signatures are divided by the per-feature SD used for standardisation before the
  Causal Match Score (cosine with the cell-type disease causal-flux vector) is computed.

## PsychENCODE / CommonMind

The DLPFC profiles analysed in the paper are available under controlled access from the PsychENCODE
Knowledge Portal (Synapse syn20821315, syn21415492) and the CommonMind Consortium (syn2755792). They cannot
be redistributed with this repository. After approval, convert the matched profiles to the layout above under
`data/psychencode/` (git-ignored) and run

```bash
python scripts/run_scz_pipeline.py --config configs/scz_psychencode.yaml
```
