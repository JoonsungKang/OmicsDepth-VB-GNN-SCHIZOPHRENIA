#!/usr/bin/env python
"""Run the multi-omic SCZ causal-discovery pipeline (main text Sections 4-5).

    python scripts/run_scz_pipeline.py --config configs/scz_demo.yaml
    python scripts/run_scz_pipeline.py --config configs/scz_psychencode.yaml   # controlled-access data

See vbgnn/pipeline.py for the ten steps and docs/DATA.md for the input formats.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbgnn.pipeline import load_config, run_pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scz_demo.yaml")
    ap.add_argument("--iters", type=int, help="override vbgnn.iters (e.g. for a smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.iters:
        cfg.setdefault("vbgnn", {})["iters"] = a.iters
    run_pipeline(cfg)
