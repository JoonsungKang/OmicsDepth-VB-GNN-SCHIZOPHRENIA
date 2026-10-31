# VB-GNN analysis report: synthetic SCZ demo cohort

> **Synthetic demo data.** The cohort was simulated with a planted causal graph (`vbgnn/demo.py`). These numbers show that the pipeline runs and can recover a planted loop; they are not findings about human biology.

- Selected edges (posterior FDR 0.05): **17** (posterior expected FDP 0.044)
- Against the planted graph: SHD 23, FDR 0.294, TPR 0.353
- Directional AUC (held-out reference): VB-GNN 0.862, NOTEARS 0.815, Correlation 0.790
- GSEA NES: Mito-Metabolic Stress 1.17, Synaptic Scaffolding 1.10, H3K27ac Remodeling 0.75
- Targeted causal flux by cell type: PV_interneuron 1.030, L23_pyramidal 0.999, Astrocyte 0.798, Microglia 0.441, Oligodendrocyte 0.327
- Causal Match Scores: Metformin -0.70, Dimethyl Fumarate -0.56, Dimethyl-aKG -0.37, Decoy_2 -0.26, Decoy_1 -0.23, Vorinostat -0.16, Haloperidol +0.04, Decoy_5 +0.10, Decoy_3 +0.29, Decoy_4 +0.33

## Top edges

| source        | target         |   posterior |   effect | in_prior   | in_reference   | true_edge   |
|:--------------|:---------------|------------:|---------:|:-----------|:---------------|:------------|
| H3K27ac_DISC1 | DISC1          |       0.999 |    0.632 | False      | True           | True        |
| GRIN2A        | SHANK3         |       0.999 |    0.674 | False      | True           | True        |
| MitoStress    | PRS_SCZ        |       0.999 |    0.609 | False      | False          | False       |
| DLG4          | SYN1           |       0.999 |    0.560 | True       | False          | True        |
| GRIN2A        | H3K27ac_GRIN2A |       0.999 |    0.367 | False      | False          | False       |
| GRIN2A        | SNP_GRIN2A     |       0.999 |    0.605 | False      | False          | False       |
| GRIN2A        | H3K9me3_GRIN2A |       0.998 |   -0.321 | False      | False          | False       |
| MitoStress    | NDUFS1         |       0.998 |   -0.445 | True       | False          | True        |
| SNP_SDHB      | SDHB           |       0.997 |   -0.566 | True       | False          | True        |
| MitoStress    | Lactate        |       0.996 |    0.453 | True       | False          | True        |
| MitoStress    | Succinate      |       0.996 |    0.271 | False      | True           | True        |
| GRIN2A        | DLG4           |       0.995 |    0.433 | True       | False          | True        |
| SDHB          | SNP_SDHB       |       0.988 |   -0.631 | False      | False          | False       |
| H3K27ac_GAPDH | GAPDH          |       0.984 |    0.561 | True       | False          | True        |
| MitoStress    | Fumarate       |       0.982 |   -0.457 | True       | False          | True        |
| Succinate     | H3K27ac_GRIN2A |       0.677 |   -0.142 | False      | True           | True        |
| Succinate     | H3K27ac_DISC1  |       0.655 |   -0.194 | False      | True           | True        |

Figures: `graph.png`, `roc.png`, `gsea.png`, `celltype_flux.png`.
