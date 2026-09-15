"""RGAT-variant Phase 1 smoke test.

Runs the exact same checks as run_phase1_smoke_test.py (see that script's
docstring for what each one verifies), just against the RGAT toy config
(experiments/configs/phase1_rgat_toy.yaml) instead of R-GCN's — its own
entry point so the RGAT variant has the same one-command gate the R-GCN
baseline does, without duplicating the check logic.

Usage (Colab or any machine with requirements.txt installed):
    python run_phase1_rgat_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_phase1_smoke_test import main

CONFIG_PATH = Path("experiments/configs/phase1_rgat_toy.yaml")

if __name__ == "__main__":
    try:
        main(CONFIG_PATH)
    except Exception as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
