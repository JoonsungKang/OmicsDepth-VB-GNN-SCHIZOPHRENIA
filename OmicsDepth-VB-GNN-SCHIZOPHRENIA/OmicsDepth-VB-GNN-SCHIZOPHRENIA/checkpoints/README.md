# Pre-trained checkpoints

`vbgnn_scz_demo.npz` — VB-GNN fitted on the training donors of the synthetic demo cohort
(`configs/scz_demo.yaml`, written by `scripts/run_scz_pipeline.py`). Load it with

```python
from vbgnn import VBGNN
model = VBGNN.load("checkpoints/vbgnn_scz_demo.npz")
model.feature_names, model.edge_probs        # posterior edge probabilities
```

The file stores the parameters, configuration, training history, `Σ`, `A_prior` and the
standardisation constants (`x_mean`, `x_std`). No checkpoint for real patient data is distributed.
