#!/usr/bin/env python
"""Synthetic benchmark of main-text Table 1 (SHD and FDR across ER-2, ER-4, SF-4).

Examples
--------
    python scripts/run_simulation.py --config configs/simulation_quick.yaml
    python scripts/run_simulation.py --config configs/simulation.yaml --jobs 16
    python scripts/run_simulation.py --config configs/simulation.yaml --export-data data/sim  # for DCDI

Outputs (in ``--out``): ``trials.csv`` (one row per method x trial),
``summary.csv`` and ``table1.md`` (mean +/- SD, the format of Table 1).
Methods whose optional dependencies are missing are reported and skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbgnn.benchmark import evaluate, simulate_trial, summarize  # noqa: E402


def build_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for g_idx, graph in enumerate(cfg["graphs"]):
        for t in range(cfg["trials"]):
            jobs.append({
                "graph": graph, "d": cfg["d"], "n": cfg["n"], "trial": t,
                "seed": int(cfg["seed"]) + 1000 * g_idx + t,
                "rho": cfg.get("rho", 0.0), "sigma2": cfg.get("sigma2", 1.0),
                "families": cfg.get("families", ["gp", "mlp", "periodic"]),
                "methods": cfg["methods"], "vbgnn": cfg.get("vbgnn", {}),
                "baselines": cfg.get("baselines", {}), "fdr_alpha": cfg.get("fdr_alpha", 0.05),
            })
    return jobs


def markdown_table(summary: pd.DataFrame, graphs: list[str], methods: list[str]) -> str:
    head = "| Method | " + " | ".join(f"{g} SHD | {g} FDR" for g in graphs) + " |"
    sep = "|---" * (1 + 2 * len(graphs)) + "|"
    rows = [head, sep]
    for m in methods:
        cells = []
        for g in graphs:
            r = summary[(summary.method == m) & (summary.graph == g)]
            if r.empty:
                cells += ["n/a", "n/a"]
            else:
                r = r.iloc[0]
                cells += [f"{r.shd_mean:.1f} ± {np.nan_to_num(r.shd_std):.1f}",
                          f"{r.fdr_mean:.2f} ± {np.nan_to_num(r.fdr_std):.2f}"]
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/simulation_quick.yaml")
    ap.add_argument("--out", default=None, help="output directory (default results/<config name>)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--methods", nargs="*", help="override the method list")
    ap.add_argument("--trials", type=int, help="override the number of trials")
    ap.add_argument("--export-data", metavar="DIR",
                    help="write every simulated data set as DIR/<graph>_trial<t>/{data1.npy,DAG1.npy} and exit")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.methods:
        cfg["methods"] = args.methods
    if args.trials:
        cfg["trials"] = args.trials
    jobs = build_jobs(cfg)

    if args.export_data:
        for j in jobs:
            B, X = simulate_trial(j["graph"], j["d"], j["n"], j["seed"], j["rho"], j["sigma2"], j["families"])
            d = os.path.join(args.export_data, f"{j['graph']}_trial{j['trial']}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "data1.npy"), X)
            np.save(os.path.join(d, "DAG1.npy"), B)
        print(f"exported {len(jobs)} data sets to {args.export_data}")
        return

    out = args.out or os.path.join("results", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out, exist_ok=True)
    print(f"{len(jobs)} replicates x {len(cfg['methods'])} methods, {args.jobs} workers -> {out}")
    records = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for i, recs in enumerate(ex.map(evaluate, jobs), 1):
                records += recs
                print(f"  [{i}/{len(jobs)}] {recs[0]['graph']} trial {recs[0]['trial']}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            records += evaluate(j)
            print(f"  [{i}/{len(jobs)}] {j['graph']} trial {j['trial']}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out, "trials.csv"), index=False)
    missing = sorted(df.loc[df.status != "ok", "method"].unique())
    for m in missing:
        msg = df.loc[(df.method == m) & (df.status != "ok"), "message"].iloc[0]
        print(f"  skipped {m}: {msg}")
    summary = summarize(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    ran = [m for m in cfg["methods"] if m not in missing]
    table = markdown_table(summary, cfg["graphs"], ran)
    note = (f"\n\nd = {cfg['d']}, n = {cfg['n']}, {cfg['trials']} Monte-Carlo trials, "
            f"rho = {cfg.get('rho', 0.0)}; mean ± SD."
            + (f" Not run (dependency missing): {', '.join(missing)}." if missing else ""))
    with open(os.path.join(out, "table1.md"), "w") as f:
        f.write(table + note + "\n")
    print("\n" + table + note)


if __name__ == "__main__":
    main()
