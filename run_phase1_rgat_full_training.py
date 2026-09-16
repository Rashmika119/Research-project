"""Trains the RGAT variant on the full FB15k-237 set.

Only run this after run_phase1_rgat_smoke_test.py has printed PASSED. This
produces `experiments/checkpoints/kg_only_baseline_rgat.pt` — the RGAT
comparison artifact alongside the R-GCN "KG-only baseline"
(experiments/checkpoints/kg_only_baseline.pt), not a quick sanity check.

Usage (Colab, GPU runtime recommended — see
experiments/configs/phase1_rgat_full.yaml):
    !python run_phase1_rgat_full_training.py

Same root-level-script reasoning as run_phase1_full_training.py: running
`training/train_kg_baseline.py` directly would put its own folder, not the
repo root, on `sys.path`, breaking its package imports.
"""

from __future__ import annotations

import yaml

from training.train_kg_baseline import run

CONFIG_PATH = "experiments/configs/phase1_rgat_full.yaml"

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    run(config)
