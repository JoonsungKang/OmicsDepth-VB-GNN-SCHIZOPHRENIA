"""Synthetic schizophrenia multi-omic demo cohort with a planted ground truth.

The real PsychENCODE / CommonMind data are controlled-access and cannot be
redistributed, so the repository ships a *synthetic* cohort with the same
structure (layers, cell types, covariates) and a **known** causal graph. It
exists to exercise every step of the pipeline end to end and to check that the
method can recover a planted mitochondrial-epigenetic feedback loop. Results on
it say nothing about human biology.

Design
------
* ``n_donors`` donors (about half SCZ), each contributing one pseudobulk profile
  per cell type (PV+ interneurons, layer II/III pyramidal neurons, astrocytes,
  microglia, oligodendrocytes): rows are ``donor x cell type``.
* Four layers: genotype (SNP dosages / PRS), metabolomics (LC-MS-like
  concentrations), epigenomics (ChIP-seq-like H3K27ac / H3K9me3 intensities at
  GRIN2A / DISC1 / GAPDH regulatory elements) and transcriptomics (snRNA-seq
  pseudobulk). Metabolites are simulated as cell-type-resolved intracellular
  levels -- a convenience of the simulation.
* Planted axis (cyclic; equilibrium solved by fixed-point iteration)::

      SNP_SDHB -> SDHB -| MitoStress -> Succinate ; MitoStress -| aKG ; Citrate -> aKG
      (Succinate - aKG) -> H3K9me3_{GRIN2A,DISC1}   (+, KDM inhibition)
      (Succinate - aKG) -| H3K27ac_{GRIN2A,DISC1}   (-)
      H3K27ac_X -> X ; H3K9me3_X -| X               (X in GRIN2A, DISC1)
      DISC1 -> GRIN2A ; SNP_GRIN2A -> GRIN2A
      DISC1 -| MitoStress ; GRIN2A -| MitoStress     (feedback loop)
      MitoStress -> Lactate (parallel marker, no downstream effect)
      GRIN2A, DISC1 -> DLG4 -> SYN1 ; GRIN2A -> SHANK3

  The metabolic-epigenetic edges are scaled by a cell-type weight (PV+ 1.0,
  L2/3 0.85, astrocytes 0.35, microglia/oligodendrocytes 0.2).
* Confounding: PMI raises Succinate and Lactate (post-mortem ischaemia), batch
  shifts all ChIP-seq signals, age raises MitoStress, sex shifts some transcripts;
  diagnosis raises MitoStress (disease effect).
* A "curated" prior (KEGG-like) contains some true edges and two wrong ones; the
  remaining true edges form the held-out reference set used for the directional
  ROC (they are never given to the model).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

CELL_TYPES = ["PV_interneuron", "L23_pyramidal", "Astrocyte", "Microglia", "Oligodendrocyte"]
CELL_WEIGHT = {"PV_interneuron": 1.0, "L23_pyramidal": 0.85, "Astrocyte": 0.35,
               "Microglia": 0.2, "Oligodendrocyte": 0.2}

LAYERS = {
    "genotype": ["SNP_SDHB", "SNP_GRIN2A", "PRS_SCZ"],
    "metabolomics": ["Succinate", "aKG", "Fumarate", "Lactate", "Citrate"],
    "epigenomics": ["H3K27ac_GRIN2A", "H3K27ac_DISC1", "H3K9me3_GRIN2A", "H3K9me3_DISC1", "H3K27ac_GAPDH"],
    "transcriptomics": ["MitoStress", "SDHB", "NDUFS1", "KDM4A", "HDAC2", "GRIN2A", "DISC1",
                        "DLG4", "SHANK3", "SYN1", "GAPDH", "GFAP"],
}
FEATURES = [f for layer in LAYERS.values() for f in layer]

# (source, target, sign, cell-type-weighted)
TRUE_EDGES = [
    ("SNP_SDHB", "SDHB", -1, False),
    ("SDHB", "MitoStress", -1, False),
    ("MitoStress", "Succinate", +1, True),
    ("MitoStress", "aKG", -1, True),
    ("Citrate", "aKG", +1, False),
    ("MitoStress", "Fumarate", -1, False),
    ("MitoStress", "Lactate", +1, False),
    ("MitoStress", "NDUFS1", -1, False),
    ("Succinate", "H3K9me3_GRIN2A", +1, True),
    ("Succinate", "H3K9me3_DISC1", +1, True),
    ("Succinate", "H3K27ac_GRIN2A", -1, True),
    ("Succinate", "H3K27ac_DISC1", -1, True),
    ("aKG", "H3K9me3_GRIN2A", -1, True),
    ("aKG", "H3K9me3_DISC1", -1, True),
    ("aKG", "H3K27ac_GRIN2A", +1, True),
    ("aKG", "H3K27ac_DISC1", +1, True),
    ("KDM4A", "H3K9me3_GRIN2A", -1, False),
    ("KDM4A", "H3K9me3_DISC1", -1, False),
    ("HDAC2", "H3K27ac_GRIN2A", -1, False),
    ("HDAC2", "H3K27ac_DISC1", -1, False),
    ("H3K27ac_GRIN2A", "GRIN2A", +1, False),
    ("H3K9me3_GRIN2A", "GRIN2A", -1, False),
    ("H3K27ac_DISC1", "DISC1", +1, False),
    ("H3K9me3_DISC1", "DISC1", -1, False),
    ("DISC1", "GRIN2A", +1, False),
    ("SNP_GRIN2A", "GRIN2A", +1, False),
    ("DISC1", "MitoStress", -1, True),
    ("GRIN2A", "MitoStress", -1, True),
    ("GRIN2A", "DLG4", +1, False),
    ("DISC1", "DLG4", +1, False),
    ("GRIN2A", "SHANK3", +1, False),
    ("DLG4", "SYN1", +1, False),
    ("H3K27ac_GAPDH", "GAPDH", +1, False),
    ("PRS_SCZ", "MitoStress", +1, False),
]

# Curated prior: well-established, pathway-database-style relations + 2 wrong ones.
PRIOR_EDGES = [
    ("SNP_SDHB", "SDHB"), ("SDHB", "MitoStress"), ("MitoStress", "Lactate"),
    ("MitoStress", "Fumarate"), ("Citrate", "aKG"), ("H3K27ac_GAPDH", "GAPDH"),
    ("GRIN2A", "DLG4"), ("DLG4", "SYN1"), ("MitoStress", "NDUFS1"),
    ("Lactate", "H3K27ac_GRIN2A"),     # wrong (lactate is not on the path)
    ("GFAP", "GRIN2A"),                 # wrong
]

GENE_SETS = {
    "Mito-Metabolic Stress": ["MitoStress", "SDHB", "NDUFS1", "Succinate", "Fumarate", "Lactate", "aKG"],
    "H3K27ac Remodeling": ["H3K27ac_GRIN2A", "H3K27ac_DISC1", "H3K9me3_GRIN2A", "H3K9me3_DISC1",
                           "KDM4A", "HDAC2"],
    "Synaptic Scaffolding": ["GRIN2A", "DISC1", "DLG4", "SHANK3", "SYN1"],
}

# Synthetic perturbation signatures: shift interventions applied to the TRUE simulator.
COMPOUNDS = {
    "Dimethyl Fumarate": ("Mitochondrial modulator & Nrf2 activator", {"MitoStress": -0.8, "Fumarate": +0.8}),
    "Vorinostat": ("Histone deacetylase (HDAC) inhibitor", {"HDAC2": -1.5}),
    "Metformin": ("Metabolic regulator & AMPK activator", {"MitoStress": -0.6, "NDUFS1": -0.5}),
    "Dimethyl-aKG": ("Cell-permeable alpha-KG ester", {"aKG": +1.5}),
    "Haloperidol": ("D2 antagonist (decoy)", {"GFAP": +0.3, "SYN1": -0.2}),
}


def _t(x):
    return np.tanh(x)


def _structural(v: dict, ex: dict, w: np.ndarray, shift: dict) -> dict:
    """One Gauss-Seidel sweep of the planted structural equations."""
    s = lambda k: shift.get(k, 0.0)  # noqa: E731
    v["SDHB"] = -0.7 * ex["SNP_SDHB_c"] + ex["e_SDHB"] + s("SDHB")
    v["KDM4A"] = ex["e_KDM4A"] + s("KDM4A")
    v["HDAC2"] = ex["e_HDAC2"] + s("HDAC2")
    v["Citrate"] = ex["e_Citrate"] + s("Citrate")
    v["MitoStress"] = (-0.6 * _t(v["SDHB"]) + 0.5 * w * ex["dx"] + 0.3 * ex["PRS_SCZ"]
                       - 0.5 * w * _t(v["DISC1"]) - 0.35 * w * _t(v["GRIN2A"])
                       + 0.25 * ex["age_c"] + ex["e_MitoStress"] + s("MitoStress"))
    v["Succinate"] = 1.0 * w * _t(v["MitoStress"]) + 0.6 * ex["pmi_c"] + ex["e_Succinate"] + s("Succinate")
    v["aKG"] = -0.5 * w * _t(v["MitoStress"]) + 0.5 * v["Citrate"] + ex["e_aKG"] + s("aKG")
    v["Fumarate"] = -0.5 * _t(v["MitoStress"]) + ex["e_Fumarate"] + s("Fumarate")
    v["Lactate"] = 0.7 * _t(v["MitoStress"]) + 0.5 * ex["pmi_c"] + ex["e_Lactate"] + s("Lactate")
    v["NDUFS1"] = -0.5 * _t(v["MitoStress"]) + ex["e_NDUFS1"] + s("NDUFS1")
    ratio = v["Succinate"] - v["aKG"]
    for g in ("GRIN2A", "DISC1"):
        v[f"H3K9me3_{g}"] = (0.9 * w * _t(ratio) - 0.4 * v["KDM4A"] + ex["batch_eff"]
                             + ex[f"e_H3K9me3_{g}"] + s(f"H3K9me3_{g}"))
        v[f"H3K27ac_{g}"] = (-0.9 * w * _t(ratio) - 0.4 * v["HDAC2"] + ex["batch_eff"]
                             + ex[f"e_H3K27ac_{g}"] + s(f"H3K27ac_{g}"))
    v["H3K27ac_GAPDH"] = ex["batch_eff"] + ex["e_H3K27ac_GAPDH"] + s("H3K27ac_GAPDH")
    v["DISC1"] = (0.9 * _t(v["H3K27ac_DISC1"]) - 0.6 * _t(v["H3K9me3_DISC1"])
                  + ex["e_DISC1"] + s("DISC1"))
    v["GRIN2A"] = (0.9 * _t(v["H3K27ac_GRIN2A"]) - 0.6 * _t(v["H3K9me3_GRIN2A"]) + 0.45 * _t(v["DISC1"])
                   + 0.4 * ex["SNP_GRIN2A_c"] + 0.2 * ex["sex"] + ex["e_GRIN2A"] + s("GRIN2A"))
    v["DLG4"] = 0.6 * _t(v["GRIN2A"]) + 0.4 * _t(v["DISC1"]) + ex["e_DLG4"] + s("DLG4")
    v["SHANK3"] = 0.7 * _t(v["GRIN2A"]) + ex["e_SHANK3"] + s("SHANK3")
    v["SYN1"] = 0.6 * _t(v["DLG4"]) + ex["e_SYN1"] + s("SYN1")
    v["GAPDH"] = 0.7 * _t(v["H3K27ac_GAPDH"]) + ex["e_GAPDH"] + s("GAPDH")
    v["GFAP"] = 0.3 * ex["sex"] + ex["e_GFAP"] + s("GFAP")
    return v


def _equilibrium(ex: dict, w: np.ndarray, shift: dict | None = None, n_iter: int = 200) -> dict:
    shift = shift or {}
    n = len(w)
    endo = [f for f in FEATURES if f not in ("SNP_SDHB", "SNP_GRIN2A", "PRS_SCZ")]
    v = {f: np.zeros(n) for f in endo}
    for _ in range(n_iter):
        old = np.column_stack([v[f] for f in endo])
        v = _structural(v, ex, w, shift)
        if np.max(np.abs(np.column_stack([v[f] for f in endo]) - old)) < 1e-10:
            break
    v["SNP_SDHB"], v["SNP_GRIN2A"], v["PRS_SCZ"] = ex["SNP_SDHB"], ex["SNP_GRIN2A"], ex["PRS_SCZ"]
    return v


def _exogenous(n_donors: int, rng: np.random.Generator, noise: float = 0.5):
    donors = np.array([f"D{i:04d}" for i in range(n_donors)])
    dx = (rng.uniform(size=n_donors) < 0.5).astype(float)
    pmi = rng.gamma(4.0, 5.0, n_donors)                  # hours, mean 20 h
    age = rng.normal(52, 14, n_donors).clip(18, 95)
    sex = (rng.uniform(size=n_donors) < 0.6).astype(float)   # 1 = male
    batch = rng.integers(0, 4, n_donors)
    snp_sdhb = rng.binomial(2, 0.3, n_donors).astype(float)
    snp_grin = rng.binomial(2, 0.4, n_donors).astype(float)
    prs = rng.normal(0, 1, n_donors) + 0.5 * dx
    batch_shift = rng.normal(0, 0.6, 4)
    rows = []
    for ct in CELL_TYPES:
        rows.append(pd.DataFrame({"donor_id": donors, "cell_type": ct, "diagnosis": dx, "pmi": pmi,
                                  "age": age, "sex": sex, "batch": batch,
                                  "SNP_SDHB": snp_sdhb, "SNP_GRIN2A": snp_grin, "PRS_SCZ": prs}))
    meta = pd.concat(rows, ignore_index=True)
    meta.insert(0, "sample_id", meta.donor_id + "__" + meta.cell_type)
    n = len(meta)
    ex = {
        "dx": meta.diagnosis.to_numpy(), "sex": meta.sex.to_numpy() - 0.6,
        "pmi_c": (meta.pmi.to_numpy() - 20.0) / 10.0, "age_c": (meta.age.to_numpy() - 52.0) / 14.0,
        "batch_eff": batch_shift[meta.batch.to_numpy()],
        "SNP_SDHB": meta.SNP_SDHB.to_numpy(), "SNP_GRIN2A": meta.SNP_GRIN2A.to_numpy(),
        "PRS_SCZ": meta.PRS_SCZ.to_numpy(),
        "SNP_SDHB_c": meta.SNP_SDHB.to_numpy() - 0.6, "SNP_GRIN2A_c": meta.SNP_GRIN2A.to_numpy() - 0.8,
    }
    for f in FEATURES:
        ex[f"e_{f}"] = rng.normal(0, noise, n)
    w = meta.cell_type.map(CELL_WEIGHT).to_numpy()
    return meta, ex, w


# Per-cell-type baseline expression (log2 scale) -- large between-cell-type differences.
_CT_OFFSET = {"PV_interneuron": {"GRIN2A": 1.0, "DLG4": 0.8, "GFAP": -2.0},
              "L23_pyramidal": {"GRIN2A": 1.2, "DLG4": 1.0, "SHANK3": 0.8, "GFAP": -2.0},
              "Astrocyte": {"GFAP": 3.0, "GRIN2A": -1.5, "SHANK3": -1.0},
              "Microglia": {"GRIN2A": -2.0, "DLG4": -1.5, "GFAP": -1.0},
              "Oligodendrocyte": {"GRIN2A": -2.0, "DLG4": -1.2, "GFAP": -0.5}}
_LAYER_BASE = {"metabolomics": 5.0, "epigenomics": 6.0, "transcriptomics": 8.0}


def simulate_scz_demo(n_donors: int = 600, seed: int = 2026, noise: float = 0.5) -> dict:
    """Simulate the demo cohort. Returns a dict of DataFrames (see module docstring)."""
    rng = np.random.default_rng(seed)
    meta, ex, w = _exogenous(n_donors, rng, noise)
    v = _equilibrium(ex, w)
    out = {"samples": meta[["sample_id", "donor_id", "cell_type", "diagnosis", "pmi", "age", "sex", "batch"]]}
    out["samples"] = out["samples"].assign(diagnosis=np.where(meta.diagnosis > 0, "SCZ", "Control"),
                                           sex=np.where(meta.sex > 0, "M", "F"),
                                           batch=meta.batch.map(lambda b: f"B{b + 1}"))
    for layer, feats in LAYERS.items():
        df = pd.DataFrame({f: v[f] for f in feats})
        if layer in _LAYER_BASE:
            for ct, offs in _CT_OFFSET.items():
                m = (meta.cell_type == ct).to_numpy()
                for f, o in offs.items():
                    if f in df:
                        df.loc[m, f] += o
            df = np.power(2.0, df + _LAYER_BASE[layer])     # raw scale; the pipeline applies log2
        df.insert(0, "sample_id", meta.sample_id)
        out[layer] = df
    out["true_edges"] = pd.DataFrame([(s, t, sg, cw) for s, t, sg, cw in TRUE_EDGES],
                                     columns=["source", "target", "sign", "cell_type_weighted"])
    out["prior_edges"] = pd.DataFrame(PRIOR_EDGES, columns=["source", "target"])
    prior = set(PRIOR_EDGES)
    out["reference_edges"] = pd.DataFrame([(s, t) for s, t, _, _ in TRUE_EDGES if (s, t) not in prior],
                                          columns=["source", "target"])
    out["compound_signatures"] = compound_signatures(meta, ex, w, rng)
    out["gene_sets"] = GENE_SETS
    return out


def compound_signatures(meta, ex, w, rng, noise: float = 0.05) -> pd.DataFrame:
    """Mean equilibrium change (treated - untreated) over SCZ samples, per compound."""
    base = _equilibrium(ex, w)
    sel = meta.diagnosis.to_numpy() > 0
    rows = []
    for name, (mech, shift) in COMPOUNDS.items():
        pert = _equilibrium(ex, w, shift)
        sig = {f: float(np.mean(pert[f][sel] - base[f][sel])) + rng.normal(0, noise) for f in FEATURES}
        rows.append({"compound": name, "mechanism": mech, **sig})
    for k in range(5):                                  # random decoys
        feats = rng.choice(FEATURES, 3, replace=False)
        shift = {f: float(rng.normal(0, 1.0)) for f in feats}
        pert = _equilibrium(ex, w, shift)
        sig = {f: float(np.mean(pert[f][sel] - base[f][sel])) + rng.normal(0, noise) for f in FEATURES}
        rows.append({"compound": f"Decoy_{k + 1}", "mechanism": "random perturbation (decoy)", **sig})
    return pd.DataFrame(rows)


def write_demo(out_dir: str, n_donors: int = 600, seed: int = 2026) -> dict:
    data = simulate_scz_demo(n_donors, seed)
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        if k == "gene_sets":
            with open(os.path.join(out_dir, "gene_sets.json"), "w") as f:
                json.dump(v, f, indent=2)
        else:
            v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False, float_format="%.6g")
    return data
