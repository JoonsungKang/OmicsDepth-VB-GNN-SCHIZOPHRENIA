#!/usr/bin/env python
"""Ablation of main-text Figure 2.

(a) FDR as the inter-entity noise correlation ``rho`` increases (ER-2).
(b) SHD as the noise variance ``sigma^2`` increases (ER-2 and SF-4).

    python scripts/run_ablation.py --config configs/ablation_quick.yaml
    python scripts/run_ablation.py --config configs/ablation.yaml --jobs 16

Writes ``rho_sweep.csv``, ``sigma2_sweep.csv`` and ``figure2.png`` to ``--out``.
The figure is drawn from these CSV files by ``scripts/make_figures.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbgnn.benchmark import evaluate  # noqa: E402


def jobs_for(cfg: dict) -> tuple[list[dict], list[dict]]:
    common = {"d": cfg["d"], "n": cfg["n"], "vbgnn": cfg.get("vbgnn", {}),
              "baselines": cfg.get("baselines", {}), "fdr_alpha": cfg.get("fdr_alpha", 0.05),
              "families": cfg.get("families", ["gp", "mlp", "periodic"])}
    a, b = [], []
    rs = cfg["rho_sweep"]
    for k, rho in enumerate(rs["values"]):
        for t in range(cfg["trials"]):
            a.append({**common, "graph": rs["graph"], "trial": t, "rho": rho,
                      "sigma2": rs.get("sigma2", 1.0), "methods": rs["methods"],
                      "seed": int(cfg["seed"]) + 100 * k + t})
    ss = cfg["sigma2_sweep"]
    for g_idx, graph in enumerate(ss["graphs"]):
        for k, s2 in enumerate(ss["values"]):
            for t in range(cfg["trials"]):
                b.append({**common, "graph": graph, "trial": t, "rho": ss.get("rho", 0.0),
                          "sigma2": s2, "methods": ss["methods"],
                          "seed": int(cfg["seed"]) + 10_000 * (g_idx + 1) + 100 * k + t})
    return a, b


def run(jobs: list[dict], n_jobs: int, label: str) -> pd.DataFrame:
    recs = []
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            for i, r in enumerate(ex.map(evaluate, jobs), 1):
                recs += r
                print(f"  {label} [{i}/{len(jobs)}]", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            recs += evaluate(j)
            print(f"  {label} [{i}/{len(jobs)}]", flush=True)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ablation_quick.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--only", choices=["rho", "sigma2"], help="run a single panel")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = args.out or os.path.join("results", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out, exist_ok=True)
    a, b = jobs_for(cfg)
    if args.only != "sigma2":
        df = run(a, args.jobs, "rho")
        df.to_csv(os.path.join(out, "rho_sweep.csv"), index=False)
        ok = df[df.status == "ok"]
        print(ok.groupby(["method", "rho"])["fdr"].agg(["mean", "std"]).round(3).unstack(0).to_string())
    if args.only != "rho":
        df = run(b, args.jobs, "sigma2")
        df.to_csv(os.path.join(out, "sigma2_sweep.csv"), index=False)
        ok = df[df.status == "ok"]
        print(ok.groupby(["graph", "method", "sigma2"])["shd"].agg(["mean", "std"]).round(2).to_string())
    from make_figures import figure2  # local import: scripts/ is on sys.path when run as a script
    figure2(out, os.path.join(out, "figure2.png"))
    print(f"wrote {out}/figure2.png")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
