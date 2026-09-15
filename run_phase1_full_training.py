"""Trains the real Phase 1 KG-only baseline on the full FB15k-237 set.

Only run this after run_phase1_smoke_test.py has printed PASSED. This
produces `experiments/checkpoints/kg_only_baseline.pt` — the actual
"KG-only baseline" artifact for the project's validation table
(README.md §12), not a quick sanity check.

Usage (Colab, GPU runtime recommended — see experiments/configs/phase1_full.yaml):
    !python run_phase1_full_training.py

A plain `python training/train_kg_baseline.py ...` would NOT work the same
way here — Python would look for the `models`/`evaluation`/`preprocessing`
packages relative to the `training/` folder instead of the repo root, and
the imports inside train_kg_baseline.py would fail. Running this root-level
script instead sidesteps that entirely, the same way run_phase0/1_smoke_test.py do.
"""

from __future__ import annotations

import sys

import yaml

from training.train_kg_baseline import run

CONFIG_PATH = "experiments/configs/phase1_full.yaml"

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run(config)
