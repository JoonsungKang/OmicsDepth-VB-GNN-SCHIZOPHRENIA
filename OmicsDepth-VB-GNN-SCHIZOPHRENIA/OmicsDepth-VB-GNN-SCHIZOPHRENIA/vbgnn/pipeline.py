"""End-to-end multi-omic analysis (main text Sections 4-5), driven by a YAML config.

Steps
-----
1.  Load and preprocess each layer, match samples, centre within cell type, standardise.
2.  Build nuisance variables (measured covariates + surrogate variables) and either pass
    them to the adversarial module (default) or regress them out.
3.  Split donors into a training and a held-out validation partition.
4.  Fit VB-GNN with the curated prior ``A_prior``.
5.  Posterior-FDR edge selection; signed direct effects.
6.  Directional ROC against the held-out reference relations (never in the prior),
    with comparison methods.
7.  Preranked GSEA on the causal-regulator score.
8.  Cell-type causal flux (Algorithm 1, line 16) and targeted Succinate -> GRIN2A/DISC1 flux.
9.  Counterfactual ``do(Succinate = control level)`` at the equilibrium of the fitted SEM.
10. Causal Match Score of compound signatures against the disease causal-flux vector.

Everything is written to ``output_dir`` (CSV, JSON, PNG and ``report.md``) and the
fitted model to ``checkpoint``.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .adversarial import surrogate_variables
from .causal import causal_flux, counterfactual_do, fit_group_models, targeted_flux
from .fdr import posterior_fdr_select
from .metrics import directional_auroc, structure_metrics
from .model import VBGNN, VBGNNConfig
from .preprocessing import LayerSpec, build_multiomic_matrix, design_matrix, residualize
from .priors import prior_from_edgelist
from .translational import causal_match_score, disease_flux_vector, preranked_gsea


def _abs(base: str, p: str | None) -> str | None:
    if p is None or os.path.isabs(p):
        return p
    return os.path.join(base, p)


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = os.path.abspath(os.path.join(os.path.dirname(path), cfg.get("root", "..")))
    return cfg


def prepare_data(cfg: dict) -> dict:
    root = cfg["_root"]
    dcfg = cfg["data"]
    layers = [LayerSpec(name=l["name"], path=_abs(root, l["path"]), log2=l.get("log2", True),
                        quantile_normalize=l.get("quantile_normalize", False),
                        max_missing=l.get("max_missing", 0.2)) for l in dcfg["layers"]]
    samples = pd.read_csv(_abs(root, dcfg["samples"])).set_index("sample_id")
    Xdf, layer_of = build_multiomic_matrix(layers, sample_ids=list(samples.index))
    samples = samples.loc[Xdf.index]
    group = samples[dcfg["group_col"]].astype(str).to_numpy() if dcfg.get("group_col") else None
    if dcfg.get("center_by_group", True) and group is not None:
        Xdf = Xdf - Xdf.groupby(group).transform("mean")
    x_mean, x_std = Xdf.mean(), Xdf.std(ddof=0).replace(0, 1.0)
    X = ((Xdf - x_mean) / x_std).to_numpy(float)
    names = list(Xdf.columns)

    cov = samples[dcfg.get("covariates", [])]
    C = design_matrix(cov)
    dx = (samples[dcfg["diagnosis_col"]].astype(str) == str(dcfg["case_label"])).to_numpy()
    n_sv = int(dcfg.get("n_surrogate_variables", 0))
    if n_sv:
        C = np.column_stack([C, surrogate_variables(X, n_sv, primary=dx, covariates=C)])
    mode = dcfg.get("nuisance_mode", "adversarial")
    if mode == "residualize" and C.shape[1]:
        X = residualize(X, C)
        X = (X - X.mean(0)) / X.std(0)
        C_model = np.zeros((len(X), 0))
    elif mode == "none":
        C_model = np.zeros((len(X), 0))
    else:
        C_model = C

    rng = np.random.default_rng(cfg.get("seed", 0))
    donors = samples[dcfg.get("donor_col", "donor_id")].astype(str).to_numpy()
    ud = np.unique(donors)
    val_donors = set(rng.choice(ud, int(round(cfg.get("val_fraction", 0.2) * len(ud))), replace=False))
    is_val = np.array([d in val_donors for d in donors])

    A_prior = (prior_from_edgelist(_abs(root, dcfg["prior_edges"]), names)
               if dcfg.get("prior_edges") else np.zeros((len(names), len(names))))
    return {"X": X, "C": C_model, "names": names, "layer_of": layer_of, "samples": samples,
            "group": group, "dx": dx, "is_val": is_val, "A_prior": A_prior,
            "x_mean": x_mean.to_numpy(), "x_std": x_std.to_numpy()}


def _pairs(path: str | None, names: list[str]) -> list[tuple[int, int]]:
    if not path or not os.path.exists(path):
        return []
    pos = {n: i for i, n in enumerate(names)}
    df = pd.read_csv(path)
    return [(pos[s], pos[t]) for s, t in zip(df.source, df.target) if s in pos and t in pos]


def _comparison_scores(name: str, X: np.ndarray) -> np.ndarray:
    if name == "Correlation":
        C = np.abs(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(C, 0)
        return C
    if name == "NOTEARS":
        from baselines import notears_linear
        return np.abs(notears_linear(X, w_threshold=0.0))
    from baselines import EXTERNAL, MissingBaseline  # noqa: F401
    from vbgnn.benchmark import run_method
    B, s = run_method(name, X)
    return s if s is not None else B


def run_pipeline(cfg: dict, verbose: bool = True) -> dict:
    t0 = time.time()
    root = cfg["_root"]
    out = _abs(root, cfg.get("output_dir", "results/scz_demo"))
    os.makedirs(out, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    D = prepare_data(cfg)
    X, C, names, is_val = D["X"], D["C"], D["names"], D["is_val"]
    log(f"[1-3] X: {X.shape[0]} samples x {X.shape[1]} features; nuisance dims {C.shape[1]}; "
        f"train {int((~is_val).sum())} / held-out {int(is_val.sum())}; prior edges {int(D['A_prior'].sum())}")

    model = VBGNN(VBGNNConfig(**cfg.get("vbgnn", {})))
    model.fit(X[~is_val], A_prior=D["A_prior"], C=C[~is_val], feature_names=names,
              X_val=X[is_val], verbose=False)
    model.x_mean, model.x_std = D["x_mean"], D["x_std"]
    ckpt = _abs(root, cfg.get("checkpoint"))
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        model.save(ckpt)
    log(f"[4] VB-GNN fitted in {time.time() - t0:.0f}s; final {model.history[-1]}")

    # 5. edges --------------------------------------------------------------------------
    P = model.edge_probs
    alpha = cfg.get("fdr_alpha", 0.05)
    sel, efdp = posterior_fdr_select(P, alpha)
    E = model.signed_effects(X[~is_val])
    pos = {n: i for i, n in enumerate(names)}
    true = _pairs(_abs(root, cfg["data"].get("true_edges")), names)
    ref = _pairs(_abs(root, cfg["data"].get("reference_edges")), names)
    rows = []
    for i, j in zip(*np.nonzero(sel)):
        rows.append({"source": names[i], "target": names[j], "posterior": P[i, j], "effect": E[i, j],
                     "in_prior": bool(D["A_prior"][i, j]), "in_reference": (i, j) in set(ref),
                     **({"true_edge": (i, j) in set(true)} if true else {})})
    edges = pd.DataFrame(rows).sort_values("posterior", ascending=False) if rows else pd.DataFrame()
    edges.to_csv(os.path.join(out, "edges.csv"), index=False)
    res = {"n_selected_edges": int(sel.sum()), "posterior_expected_fdp": efdp, "fdr_alpha": alpha}
    if true:
        B = np.zeros_like(P)
        for i, j in true:
            B[i, j] = 1
        res["vs_ground_truth"] = structure_metrics(B, sel)
    log(f"[5] {res['n_selected_edges']} edges at posterior FDR {alpha} (E[FDP] = {efdp:.3f})"
        + (f"; vs planted truth {res['vs_ground_truth']}" if true else ""))

    # 6. directional ROC on the held-out reference set ------------------------------------
    roc = {}
    if ref:
        auc, fpr, tpr = directional_auroc(P, ref, n_random_negatives=3 * len(ref), seed=cfg.get("seed", 0))
        roc["VB-GNN"] = (auc, fpr, tpr)
        for name in cfg.get("comparison_methods", []):
            try:
                S = _comparison_scores(name, X[~is_val])
            except Exception as e:  # optional dependency missing
                log(f"    comparison {name} skipped: {str(e)[:80]}")
                continue
            roc[name] = directional_auroc(S, ref, n_random_negatives=3 * len(ref), seed=cfg.get("seed", 0))
        pd.DataFrame([{"method": k, "auc": v[0]} for k, v in roc.items()]).to_csv(
            os.path.join(out, "directional_auc.csv"), index=False)
        res["directional_auc"] = {k: v[0] for k, v in roc.items()}
        res["heldout_nll"] = {"vbgnn": model.score_nll(X[is_val])}
        log(f"[6] directional AUC vs {len(ref)} held-out reference relations: "
            + ", ".join(f"{k} {v[0]:.3f}" for k, v in roc.items()))

    # 7. GSEA on the causal-regulator score -------------------------------------------------
    regulator = (P * np.abs(E)).sum(1)
    reg_scores = dict(zip(names, regulator))
    gsea = []
    gs_path = _abs(root, cfg["data"].get("gene_sets"))
    if gs_path and os.path.exists(gs_path):
        with open(gs_path) as f:
            gsets = json.load(f)
        gsea = preranked_gsea(reg_scores, gsets, n_perm=cfg.get("gsea_permutations", 1000),
                              seed=cfg.get("seed", 0))
        pd.DataFrame(gsea).to_csv(os.path.join(out, "gsea.csv"), index=False)
        res["gsea"] = {r["set"]: {"nes": r["nes"], "pval": r["pval"]} for r in gsea}
        log("[7] GSEA NES: " + ", ".join(f"{r['set']} {r['nes']:.2f} (p={r['pval']:.3f})" for r in gsea))
    pd.DataFrame({"feature": names, "regulator_score": regulator,
                  "layer": [D["layer_of"][n] for n in names]}).sort_values(
        "regulator_score", ascending=False).to_csv(os.path.join(out, "regulators.csv"), index=False)

    # 8. cell-type causal flux --------------------------------------------------------------
    ax = cfg.get("axis", {})
    src = ax.get("source", "Succinate")
    tgts = [t for t in ax.get("targets", ["GRIN2A", "DISC1"]) if t in pos]
    flux_rows = []
    group_models = {}
    if D["group"] is not None and src in pos:
        tr = ~is_val
        group_models = fit_group_models(model, X[tr], D["group"][tr], D["A_prior"], C[tr],
                                        iters=ax.get("group_iters", 1500))
        Fg, tf = {}, {}
        for g, gm in group_models.items():
            m = D["group"] == g
            Fg[g] = causal_flux(gm, X[m])[0]
            tf[g] = targeted_flux(gm, X[m], np.zeros(m.sum()), pos[src], [pos[t] for t in tgts],
                                  delta=ax.get("delta", 1.0), max_iter=300)[0]
        for g in Fg:
            flux_rows.append({"group": g, "targeted_flux": tf[g], "jacobian_flux_total": float(np.linalg.norm(Fg[g])),
                              **{f"F_{n}": Fg[g][pos[n]] for n in [src, *tgts]}})
        flux = pd.DataFrame(flux_rows).sort_values("targeted_flux", ascending=False)
        flux.to_csv(os.path.join(out, "celltype_flux.csv"), index=False)
        res["celltype_targeted_flux"] = dict(zip(flux.group, flux.targeted_flux))
        log(f"[8] targeted flux {src} -> {'/'.join(tgts)}: "
            + ", ".join(f"{g} {v:.3f}" for g, v in zip(flux.group, flux.targeted_flux)))

    # 9. counterfactual do(source = control level) ------------------------------------------
    cf_rows = []
    if src in pos:
        dx = D["dx"]
        grp = D["group"] if D["group"] is not None else np.array(["all"] * len(X))
        focus = ax.get("counterfactual_groups") or list(np.unique(grp))
        report = [f for f in ax.get("report", tgts) if f in pos]
        for g in focus:
            case, ctrl = (grp == g) & dx, (grp == g) & ~dx
            if case.sum() == 0 or ctrl.sum() == 0:
                continue
            level = float(X[ctrl, pos[src]].mean())
            Xcf, info = counterfactual_do(group_models.get(g, model), X[case], {pos[src]: level}, max_iter=500)
            for f in report:
                k = pos[f]
                gap = X[ctrl, k].mean() - X[case, k].mean()
                shift = Xcf[:, k].mean() - X[case, k].mean()
                cf_rows.append({"group": g, "feature": f, "case_mean": X[case, k].mean(),
                                "control_mean": X[ctrl, k].mean(), "counterfactual_mean": Xcf[:, k].mean(),
                                "fraction_of_gap_recovered": shift / gap if abs(gap) > 1e-8 else np.nan,
                                "converged": info["converged"]})
        cf = pd.DataFrame(cf_rows)
        cf.to_csv(os.path.join(out, "counterfactual.csv"), index=False)
        if len(cf):
            res["counterfactual"] = {f"{r.group}:{r.feature}": round(float(r.fraction_of_gap_recovered), 3)
                                     for r in cf.itertuples()}
            log(f"[9] do({src} = control mean), fraction of case-control gap recovered:\n"
                + cf.pivot(index="feature", columns="group", values="fraction_of_gap_recovered").round(2).to_string())

    # 10. Causal Match Score ----------------------------------------------------------------
    sig_path = _abs(root, cfg["data"].get("compound_signatures"))
    if sig_path and os.path.exists(sig_path) and D["group"] is not None:
        sigs = pd.read_csv(sig_path)
        g = ax.get("drug_group", "PV_interneuron")
        m = D["group"] == g
        Fg = causal_flux(group_models.get(g, model), X[m])[0]
        v = disease_flux_vector(X[m], D["dx"][m], Fg)
        rows = []
        for r in sigs.itertuples(index=False):
            r = r._asdict()
            s = np.array([r.get(n, np.nan) for n in names], float) / D["x_std"]   # to standardised units
            rows.append({"compound": r["compound"], "mechanism": r.get("mechanism", ""),
                         "causal_match_score": causal_match_score(s, v)})
        drugs = pd.DataFrame(rows).sort_values("causal_match_score")
        drugs.to_csv(os.path.join(out, "causal_match_scores.csv"), index=False)
        res["causal_match_scores"] = dict(zip(drugs.compound, drugs.causal_match_score.round(3)))
        log("[10] Causal Match Score (" + g + "):\n" + drugs.to_string(index=False))

    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    from .plots import pipeline_figures
    pipeline_figures(out, model, names, P, sel, E, roc, gsea, flux_rows, cfg)
    _write_report(out, cfg, res, edges)
    log(f"done in {res['runtime_seconds']}s -> {out}")
    return {"model": model, "data": D, "results": res, "edges": edges, "output_dir": out}


def _write_report(out: str, cfg: dict, res: dict, edges: pd.DataFrame) -> None:
    lines = [f"# VB-GNN analysis report: {cfg.get('name', 'analysis')}", ""]
    if cfg.get("synthetic", False):
        lines += ["> **Synthetic demo data.** The cohort was simulated with a planted causal graph "
                  "(`vbgnn/demo.py`). These numbers show that the pipeline runs and can recover a planted "
                  "loop; they are not findings about human biology.", ""]
    lines += [f"- Selected edges (posterior FDR {res['fdr_alpha']}): **{res['n_selected_edges']}** "
              f"(posterior expected FDP {res['posterior_expected_fdp']:.3f})"]
    if "vs_ground_truth" in res:
        g = res["vs_ground_truth"]
        lines.append(f"- Against the planted graph: SHD {g['shd']}, FDR {g['fdr']:.3f}, TPR {g['tpr']:.3f}")
    if "directional_auc" in res:
        lines.append("- Directional AUC (held-out reference): "
                     + ", ".join(f"{k} {v:.3f}" for k, v in res["directional_auc"].items()))
    if "gsea" in res:
        lines.append("- GSEA NES: " + ", ".join(f"{k} {v['nes']:.2f}" for k, v in res["gsea"].items()))
    if "celltype_targeted_flux" in res:
        lines.append("- Targeted causal flux by cell type: "
                     + ", ".join(f"{k} {v:.3f}" for k, v in res["celltype_targeted_flux"].items()))
    if "causal_match_scores" in res:
        lines.append("- Causal Match Scores: "
                     + ", ".join(f"{k} {v:+.2f}" for k, v in res["causal_match_scores"].items()))
    lines += ["", "## Top edges", ""]
    if len(edges):
        try:
            lines.append(edges.head(40).to_markdown(index=False, floatfmt=".3f"))
        except ImportError:  # `tabulate` not installed
            lines += ["```", edges.head(40).to_string(index=False, float_format="%.3f"), "```"]
    lines += ["", "Figures: `graph.png`, `roc.png`, `gsea.png`, `celltype_flux.png`.", ""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines))
