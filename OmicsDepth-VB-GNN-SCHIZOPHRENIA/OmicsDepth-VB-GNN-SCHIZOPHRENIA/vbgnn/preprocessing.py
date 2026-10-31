"""Multi-omic preprocessing (main text Section 2.3/4.1, SI Section 2.3).

Each molecular layer is processed separately -- QC filtering, optional log2
transform, optional quantile normalisation, imputation, standardisation -- then
layers are matched on sample IDs and concatenated into ``X = [X^(1), ..., X^(L)]``.
Measured covariates are either regressed out or passed on as nuisance variables
for the adversarial module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LayerSpec:
    name: str
    path: str
    log2: bool = True
    pseudocount: float = 1.0
    quantile_normalize: bool = True
    max_missing: float = 0.2
    min_variance: float = 1e-8


def qc_filter(df: pd.DataFrame, max_missing: float = 0.2, min_variance: float = 1e-8) -> pd.DataFrame:
    """Drop features with too many missing values or (near-)zero variance."""
    keep = df.isna().mean() <= max_missing
    df = df.loc[:, keep]
    var = df.var(skipna=True)
    return df.loc[:, var > min_variance]


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalise samples (rows) to a common distribution within a layer."""
    arr = df.to_numpy(float)
    ranks = np.argsort(np.argsort(arr, axis=1), axis=1)
    mean_sorted = np.sort(arr, axis=1).mean(0)
    return pd.DataFrame(mean_sorted[ranks], index=df.index, columns=df.columns)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1.0)


def design_matrix(cov: pd.DataFrame) -> np.ndarray:
    """One-hot encode categorical covariates, standardise numeric ones."""
    parts = []
    for c in cov.columns:
        col = cov[c]
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            dummies = pd.get_dummies(col.astype(str), prefix=c, drop_first=True).astype(float)
            parts.append(dummies)
        else:
            v = col.astype(float)
            parts.append(((v - v.mean()) / (v.std(ddof=0) or 1.0)).to_frame(c))
    return pd.concat(parts, axis=1).fillna(0.0).to_numpy(float) if parts else np.zeros((len(cov), 0))


def residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove the linear effect of covariates ``C`` from every column of ``X``."""
    D = np.column_stack([np.ones(len(X)), C])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef + X.mean(0)


def preprocess_layer(df: pd.DataFrame, spec: LayerSpec) -> pd.DataFrame:
    df = qc_filter(df.apply(pd.to_numeric, errors="coerce"), spec.max_missing, spec.min_variance)
    if spec.log2:
        df = np.log2(df.clip(lower=0) + spec.pseudocount)
    df = df.fillna(df.median())
    if spec.quantile_normalize and df.shape[1] > 1:
        df = quantile_normalize(df)
    return df


def build_multiomic_matrix(layers: list[LayerSpec], sample_ids: list[str] | None = None,
                           index_col: str = "sample_id") -> tuple[pd.DataFrame, dict[str, str]]:
    """Load, preprocess and concatenate layers on shared samples.

    Returns the concatenated (not yet standardised) frame and a ``feature -> layer`` map.
    """
    frames, layer_of = [], {}
    for spec in layers:
        df = pd.read_csv(spec.path, index_col=index_col)
        df = preprocess_layer(df, spec)
        frames.append(df)
        layer_of.update({c: spec.name for c in df.columns})
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    if sample_ids is not None:
        common = pd.Index([s for s in sample_ids if s in set(common)])
    X = pd.concat([f.loc[common] for f in frames], axis=1)
    return X, layer_of


def pseudobulk(counts: pd.DataFrame, cell_meta: pd.DataFrame, sample_col: str = "donor_id",
               celltype_col: str = "cell_type") -> pd.DataFrame:
    """Aggregate single-nucleus counts (cells x genes) to (donor, cell type) pseudobulk sums."""
    keys = cell_meta.loc[counts.index, [sample_col, celltype_col]].astype(str)
    grouped = counts.groupby([keys[sample_col], keys[celltype_col]]).sum()
    grouped.index = [f"{a}__{b}" for a, b in grouped.index]
    return grouped
