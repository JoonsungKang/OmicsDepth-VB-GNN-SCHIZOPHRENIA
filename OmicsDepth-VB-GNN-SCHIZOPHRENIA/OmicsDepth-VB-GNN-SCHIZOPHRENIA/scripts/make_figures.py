#!/usr/bin/env python
"""Draw the benchmark figures from the CSV files written by the run scripts.

    python scripts/make_figures.py --ablation results/ablation_quick
    python scripts/make_figures.py --simulation results/simulation_quick

Every method keeps the same colour in every panel (colour follows the method,
never its rank); identity is also carried by marker shape and a legend.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (validated reference palette, light mode).
METHOD_STYLE = {
    "VB-GNN":         {"color": "#2a78d6", "marker": "o", "lw": 2.4, "z": 5},
    "GraN-DAG":       {"color": "#eb6834", "marker": "s", "lw": 1.6, "z": 3},
    "DCDI":           {"color": "#1baf7a", "marker": "^", "lw": 1.6, "z": 3},
    "DAG-GNN":        {"color": "#eda100", "marker": "D", "lw": 1.6, "z": 3},
    "VB-GNN-Shannon": {"color": "#e87ba4", "marker": "v", "lw": 1.6, "z": 4},
    "NOTEARS":        {"color": "#008300", "marker": "P", "lw": 1.6, "z": 3},
    "PC":             {"color": "#6b6a64", "marker": "X", "lw": 1.6, "z": 3},
    "VB-GNN-NoReg":   {"color": "#9a9890", "marker": "*", "lw": 1.6, "z": 3},
}
INK, MUTED, GRID = "#1f1e1b", "#6b6a64", "#e4e2dc"


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _lines(ax, df, x, y):
    ok = df[df.status == "ok"]
    for method, g in ok.groupby("method"):
        st = METHOD_STYLE.get(method, {"color": MUTED, "marker": "o", "lw": 1.5, "z": 2})
        agg = g.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.plot(agg[x], agg["mean"], color=st["color"], marker=st["marker"], ms=7,
                lw=st["lw"], zorder=st["z"], label=method, markeredgecolor="white", markeredgewidth=1.5)
        ax.fill_between(agg[x], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0),
                        color=st["color"], alpha=0.12, lw=0, zorder=1)


def figure2(result_dir: str, path: str) -> None:
    rho_csv = os.path.join(result_dir, "rho_sweep.csv")
    s2_csv = os.path.join(result_dir, "sigma2_sweep.csv")
    panels = [p for p in (rho_csv, s2_csv) if os.path.exists(p)]
    if not panels:
        raise FileNotFoundError(f"no sweep CSV in {result_dir}")
    s2 = pd.read_csv(s2_csv) if os.path.exists(s2_csv) else None
    n_s2 = s2.graph.nunique() if s2 is not None else 0
    ncol = (1 if os.path.exists(rho_csv) else 0) + n_s2
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.8), squeeze=False)
    axes = list(axes[0])
    if os.path.exists(rho_csv):
        ax = axes.pop(0)
        _lines(ax, pd.read_csv(rho_csv), "rho", "fdr")
        _style_axes(ax, "(a) FDR vs inter-entity covariance", "Noise correlation ρ", "False discovery rate")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
    if s2 is not None:
        for (graph, g), ax in zip(s2.groupby("graph"), axes):
            _lines(ax, g, "sigma2", "shd")
            _style_axes(ax, f"(b) SHD vs noise variance ({graph})", "Noise variance σ²",
                        "Structural Hamming distance")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def simulation_figure(result_dir: str, path: str) -> None:
    """Grouped bars of mean SHD and FDR per graph family (Table 1 as a figure)."""
    df = pd.read_csv(os.path.join(result_dir, "trials.csv"))
    ok = df[df.status == "ok"]
    graphs = list(dict.fromkeys(ok.graph))
    methods = [m for m in METHOD_STYLE if m in set(ok.method)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.8 / max(1, len(methods))
    for ax, metric, label in zip(axes, ["shd", "fdr"], ["Structural Hamming distance", "False discovery rate"]):
        for k, m in enumerate(methods):
            agg = ok[ok.method == m].groupby("graph")[metric].agg(["mean", "std"]).reindex(graphs)
            xs = [i + (k - (len(methods) - 1) / 2) * width for i in range(len(graphs))]
            ax.bar(xs, agg["mean"], width * 0.9, yerr=agg["std"].fillna(0), label=m,
                   color=METHOD_STYLE[m]["color"], edgecolor="white", lw=1,
                   error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.set_xticks(range(len(graphs)), graphs)
        _style_axes(ax, label, "Graph family", label)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", help="directory with rho_sweep.csv / sigma2_sweep.csv")
    ap.add_argument("--simulation", help="directory with trials.csv")
    a = ap.parse_args(argv)
    if a.ablation:
        figure2(a.ablation, os.path.join(a.ablation, "figure2.png"))
        print(f"wrote {a.ablation}/figure2.png")
    if a.simulation:
        simulation_figure(a.simulation, os.path.join(a.simulation, "table1.png"))
        print(f"wrote {a.simulation}/table1.png")


if __name__ == "__main__":
    main()
