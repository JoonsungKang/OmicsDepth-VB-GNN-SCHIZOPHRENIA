"""Adapters to third-party implementations of DAG-GNN, GraN-DAG and DCDI.

These methods are run with their *original* code bases, which are PyTorch-based
and therefore optional dependencies:

* DAG-GNN and GraN-DAG -- ``pip install gcastle torch`` (Huawei Noah's Ark gCastle)
* DCDI -- clone https://github.com/slachapelle/dcdi and put it on ``PYTHONPATH``

Each adapter returns a binary adjacency ``B[i, j] = 1`` for ``i -> j`` or raises
:class:`MissingBaseline` with installation instructions.
"""
from __future__ import annotations

import numpy as np


class MissingBaseline(RuntimeError):
    pass


def dag_gnn(X: np.ndarray, **kw) -> np.ndarray:
    try:
        from castle.algorithms import DAG_GNN
    except Exception as e:  # pragma: no cover
        raise MissingBaseline("DAG-GNN needs `pip install gcastle torch`") from e
    model = DAG_GNN(**kw)
    model.learn(X)
    return (np.asarray(model.causal_matrix) != 0).astype(float)


def gran_dag(X: np.ndarray, **kw) -> np.ndarray:
    try:
        from castle.algorithms import GraNDAG
    except Exception as e:  # pragma: no cover
        raise MissingBaseline("GraN-DAG needs `pip install gcastle torch`") from e
    model = GraNDAG(input_dim=X.shape[1], **kw)
    model.learn(X)
    return (np.asarray(model.causal_matrix) != 0).astype(float)


def dcdi(X: np.ndarray, tag: str | None = None, **kw) -> np.ndarray:  # pragma: no cover
    """DCDI is run with its official code (https://github.com/slachapelle/dcdi).

    ``scripts/run_simulation.py --export-data DIR`` writes every simulated data set
    as ``DIR/<graph>_trial<t>/data1.npy`` (DCDI's input format). Run DCDI on each
    folder, collect the learned adjacency matrices as
    ``$DCDI_RESULTS/<graph>_trial<t>.npy`` and the benchmark will pick them up.
    """
    import os

    root = os.environ.get("DCDI_RESULTS")
    path = os.path.join(root, f"{tag}.npy") if root and tag else None
    if not path or not os.path.exists(path):
        raise MissingBaseline("DCDI: run the official code on the exported data and set DCDI_RESULTS")
    return (np.load(path) != 0).astype(float)


REGISTRY = {"DAG-GNN": dag_gnn, "GraN-DAG": gran_dag, "DCDI": dcdi}
