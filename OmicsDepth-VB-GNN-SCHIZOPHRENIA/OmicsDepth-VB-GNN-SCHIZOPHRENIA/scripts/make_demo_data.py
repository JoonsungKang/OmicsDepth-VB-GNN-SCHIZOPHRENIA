#!/usr/bin/env python
"""Write the synthetic schizophrenia demo cohort to ``data/demo/`` (see vbgnn/demo.py).

    python scripts/make_demo_data.py --out data/demo --donors 600 --seed 2026
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vbgnn.demo import write_demo  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--donors", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    data = write_demo(a.out, a.donors, a.seed)
    n = len(data["samples"])
    print(f"wrote {n} samples ({a.donors} donors x {n // a.donors} cell types), "
          f"{sum(len(v.columns) - 1 for k, v in data.items() if k in ('genotype', 'metabolomics', 'epigenomics', 'transcriptomics'))} features, "
          f"{len(data['true_edges'])} true edges, {len(data['reference_edges'])} held-out reference edges -> {a.out}")
