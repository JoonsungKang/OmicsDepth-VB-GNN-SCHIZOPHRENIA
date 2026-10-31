"""Figures of the multi-omic pipeline (main text Figures 3b, 4a, 4b; cell-type flux)."""
from __future__ import annotations

import os

import numpy as np

INK, MUTED, GRID, SURFACE = "#1f1e1b", "#6b6a64", "#e4e2dc", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
METHOD_COLORS = {"VB-GNN": BLUE, "GraN-DAG": ORANGE, "DCDI": AQUA, "DAG-GNN": YELLOW,
                 "NOTEARS": GREEN, "Correlation": MUTED}
LAYER_COLORS = {"genotype": MUTED, "metabolomics": ORANGE, "epigenomics": BLUE, "transcriptomics": GREEN}


def _axes(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def axis_graph(path, names, P, sel, E, focus, layer_of=None, title="VB-GNN inferred axis"):
    """Directed graph of the selected edges among ``focus`` nodes (circular layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    idx = [names.index(f) for f in focus if f in names]
    k = len(idx)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, k, endpoint=False)
    xy = {i: (np.cos(a), np.sin(a)) for i, a in zip(idx, ang)}
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for i in idx:
        for j in idx:
            if i != j and sel[i, j]:
                color = BLUE if E[i, j] >= 0 else ORANGE
                rad = 0.15 if sel[j, i] else 0.05
                ax.add_patch(FancyArrowPatch(xy[i], xy[j], arrowstyle="-|>", mutation_scale=14,
                                             color=color, lw=0.8 + 2.4 * P[i, j], alpha=0.85,
                                             connectionstyle=f"arc3,rad={rad}", shrinkA=22, shrinkB=22))
    for i in idx:
        layer = (layer_of or {}).get(names[i], "")
        ax.scatter(*xy[i], s=1500, color=SURFACE, edgecolor=LAYER_COLORS.get(layer, MUTED), lw=2.2, zorder=3)
        ax.text(*xy[i], names[i].replace("_", "\n", 1), ha="center", va="center", fontsize=6.5, color=INK, zorder=4)
    ax.plot([], [], color=BLUE, lw=2, label="positive effect")
    ax.plot([], [], color=ORANGE, lw=2, label="negative effect")
    for layer, c in LAYER_COLORS.items():
        ax.scatter([], [], s=60, color=SURFACE, edgecolor=c, lw=2, label=layer)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pipeline_figures(out, model, names, P, sel, E, roc, gsea, flux_rows, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .preprocessing import LayerSpec  # noqa: F401  (layer names come from the config)

    layer_of = {}
    try:
        from .pipeline import _abs
        import pandas as pd
        for l in cfg["data"]["layers"]:
            cols = pd.read_csv(_abs(cfg["_root"], l["path"]), nrows=0).columns[1:]
            layer_of.update({c: l["name"] for c in cols})
    except Exception:
        pass
    focus = cfg.get("axis", {}).get("graph_nodes") or names
    axis_graph(os.path.join(out, "graph.png"), names, P, sel, E, focus, layer_of)

    if roc:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for name, (auc, fpr, tpr) in roc.items():
            ax.plot(fpr, tpr, color=METHOD_COLORS.get(name, MUTED), lw=2.4 if name == "VB-GNN" else 1.6,
                    ls="-" if name == "VB-GNN" else "--", label=f"{name} (AUC = {auc:.2f})")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls=":")
        _axes(ax, "(a) Directional ROC, held-out relations", "False positive rate", "True positive rate")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(True, color=GRID, lw=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "roc.png"), dpi=200)
        plt.close(fig)

    if gsea:
        fig, ax = plt.subplots(figsize=(5.2, 2.2 + 0.35 * len(gsea)))
        rows = sorted(gsea, key=lambda r: np.nan_to_num(r["nes"]))
        y = np.arange(len(rows))
        ax.barh(y, [r["nes"] for r in rows], color=BLUE, height=0.55, edgecolor=SURFACE)
        for yi, r in zip(y, rows):
            ax.text(r["nes"], yi, f"  {r['nes']:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [r["set"] for r in rows])
        ax.axvline(0, color=GRID, lw=1)
        _axes(ax, "(b) Pathway enrichment of causal regulators", "Normalized enrichment score (NES)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "gsea.png"), dpi=200)
        plt.close(fig)

    if flux_rows:
        rows = sorted(flux_rows, key=lambda r: r["targeted_flux"])
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        y = np.arange(len(rows))
        ax.barh(y, [r["targeted_flux"] for r in rows], color=BLUE, height=0.55, edgecolor=SURFACE)
        ax.set_yticks(y, [r["group"] for r in rows])
        src = cfg.get("axis", {}).get("source", "Succinate")
        tg = "/".join(cfg.get("axis", {}).get("targets", ["GRIN2A", "DISC1"]))
        _axes(ax, f"Causal flux {src} → {tg}", "Equilibrium response per unit shift")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "celltype_flux.png"), dpi=200)
        plt.close(fig)
