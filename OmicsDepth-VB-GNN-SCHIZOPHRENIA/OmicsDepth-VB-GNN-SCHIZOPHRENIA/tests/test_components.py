"""FDR rules, metrics, simulation, model fitting, counterfactuals and the demo pipeline."""
import os

import numpy as np
import pytest

from vbgnn import (VBGNN, VBGNNConfig, benjamini_hochberg, counterfactual_do, edge_auroc,
                   posterior_fdr_select, structure_metrics)
from vbgnn.relaxation import concrete_sample
from vbgnn.simulation import make_graph, simulate_anm, topological_order

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_posterior_fdr_rule():
    P = np.array([[0, .99, .98, .10], [.02, 0, .95, .60], [.01, .01, 0, .9], [.0, .0, .0, 0]])
    sel, efdp = posterior_fdr_select(P, 0.05)
    chosen = P[sel]
    assert efdp <= 0.05 and np.isclose(efdp, np.mean(1 - chosen))
    # adding the next-best edge would break the rule
    nxt = np.sort(P[~sel & ~np.eye(4, dtype=bool)])[-1]
    assert np.mean(1 - np.append(chosen, nxt)) > 0.05


def test_bh_and_by():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    assert benjamini_hochberg(p, 0.05).sum() == 2
    assert benjamini_hochberg(p, 0.05, "arbitrary").sum() <= 2


def test_structure_metrics_conventions():
    B = np.zeros((3, 3)); B[0, 1] = B[1, 2] = 1
    E = np.zeros((3, 3)); E[1, 0] = E[1, 2] = E[0, 2] = 1   # one reversed, one correct, one extra
    m = structure_metrics(B, E)
    assert m == {"shd": 2, "fdr": 2 / 3, "tpr": 0.5, "nnz": 3, "reversed": 1}
    assert edge_auroc(B, B + 0.0) == 1.0


@pytest.mark.parametrize("kind,k", [("ER-2", 2), ("ER-4", 4), ("SF-4", 4)])
def test_simulated_graphs_are_dags_with_expected_density(kind, k):
    rng = np.random.default_rng(0)
    B = make_graph(kind, 100, rng)
    topological_order(B)                                   # raises if cyclic
    assert 0.7 * k * 100 < B.sum() < 1.3 * k * 100
    X = simulate_anm(B, 200, rng, rho=0.5)
    assert X.shape == (200, 100) and np.allclose(X.mean(0), 0, atol=1e-8)


def test_concrete_rounding_property():
    rng = np.random.default_rng(1)
    P = np.full((200, 200), 0.3)
    Z, _ = concrete_sample(P, 0.7, rng.logistic(size=P.shape))
    assert abs((Z > 0.5).mean() - 0.3) < 0.01


def test_vbgnn_recovers_a_small_chain():
    rng = np.random.default_rng(0)
    n = 1500
    x0 = rng.normal(size=n)
    x1 = np.tanh(2 * x0) + 0.3 * rng.normal(size=n)
    x2 = np.sin(2 * x1) + 0.3 * rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([x0, x1, x2, x3])
    X = (X - X.mean(0)) / X.std(0)
    m = VBGNN(VBGNNConfig(iters=1500, seed=0)).fit(X)
    P = m.edge_probs
    assert P[0, 1] > 0.9 and P[1, 2] > 0.9
    assert P[0, 1] > P[1, 0] and P[1, 2] > P[2, 1]
    assert P[3].max() < 0.5 and P[:, 3].max() < 0.5


def test_save_load_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(200, 3))
    m = VBGNN(VBGNNConfig(iters=50)).fit(X, feature_names=["a", "b", "c"])
    m.save(tmp_path / "m.npz")
    m2 = VBGNN.load(tmp_path / "m.npz")
    assert np.allclose(m.predict(X), m2.predict(X)) and m2.feature_names == ["a", "b", "c"]


def test_counterfactual_equilibrium_on_a_cycle():
    """With a known 2-node feedback loop, do(x0 = v) must give the fixed point of x1."""
    class Linear:
        def predict(self, X, Z=None):
            out = np.zeros_like(X)
            out[:, 0] = 0.5 * X[:, 1]
            out[:, 1] = 0.8 * X[:, 0]
            return out
    X = np.zeros((4, 2))
    Xcf, info = counterfactual_do(Linear(), X, {0: 2.0}, abduct=False)
    assert info["converged"] and np.allclose(Xcf[:, 1], 1.6)
    Xeq, _ = counterfactual_do(Linear(), np.ones((1, 2)), {}, abduct=False)
    assert np.allclose(Xeq, 0, atol=1e-5)                  # unique equilibrium of the loop


def test_demo_pipeline_smoke(tmp_path):
    from vbgnn.pipeline import load_config, run_pipeline
    cfg = load_config(os.path.join(ROOT, "configs", "scz_demo.yaml"))
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["checkpoint"] = str(tmp_path / "ckpt.npz")
    cfg["vbgnn"]["iters"] = 150
    cfg["axis"]["group_iters"] = 30
    cfg["gsea_permutations"] = 50
    cfg["comparison_methods"] = ["Correlation"]
    r = run_pipeline(cfg, verbose=False)
    for f in ("edges.csv", "summary.json", "report.md", "roc.png", "graph.png",
              "celltype_flux.csv", "counterfactual.csv", "causal_match_scores.csv"):
        assert os.path.exists(tmp_path / "out" / f), f
    assert 0.0 <= r["results"]["directional_auc"]["VB-GNN"] <= 1.0
