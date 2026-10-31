"""OmicsDepth VB-GNN: dependency-aware multi-omic causal discovery.

Reference implementation accompanying
"A Mitochondrial-Epigenetic Feedback Loop Linked to Synaptic Gene Repression in
Schizophrenia Uncovered by Multi-Omic Causal Discovery" (Jeon, Chun, Yun & Kang).
"""
from .causal import causal_flux, counterfactual_do, targeted_flux
from .fdr import benjamini_hochberg, posterior_fdr_select
from .metrics import directional_auroc, edge_auroc, structure_metrics
from .model import VBGNN, VBGNNConfig, shrinkage_covariance
from .regularizers import egs_struct_loss, shannon_struct_loss, structural_diversity

__version__ = "1.0.0"

__all__ = [
    "VBGNN", "VBGNNConfig", "shrinkage_covariance",
    "egs_struct_loss", "shannon_struct_loss", "structural_diversity",
    "posterior_fdr_select", "benjamini_hochberg",
    "structure_metrics", "edge_auroc", "directional_auroc",
    "causal_flux", "counterfactual_do", "targeted_flux",
]
