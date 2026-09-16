"""Phase 1 smoke test — run this before training on the full FB15k-237 set.

Usage (Colab or any machine with requirements.txt installed):
    python run_phase1_smoke_test.py

What this checks (mirrors the Phase 1 gate in CLAUDE.md):
  1. The full Phase 1 model (Encoder Phase 1 + DistMult) trains on the small
     toy subset without crashing, without producing NaN/inf loss, and with
     the loss trending down over a handful of epochs.
  2. Validation filtered MRR/Hits@K can actually be computed without error
     (the numbers themselves aren't meaningful on random toy data with a
     tiny, randomly-initialized model — this only proves the evaluation
     code path works).
  3. The trained model checkpoint saves and reloads correctly (a fresh model
     loaded from the checkpoint reproduces identical parameters).

Exits non-zero with a clear message if any check fails. Only once this
passes should training move on to the full FB15k-237 training set (a
separate config/run, not this script).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from training.model_factory import build_model
from training.train_kg_baseline import run

CONFIG_PATH = Path("experiments/configs/phase1_toy.yaml")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"Phase 1 smoke test FAILED: {message}")
    print(f"  [ok] {message}")


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("--- Training on toy subset ---")
    result = run(config)
    history = result["history"]
    model = result["model"]

    print("\n--- Checks ---")
    losses = [row["loss"] for row in history]
    _check(
        all(l == l and l not in (float("inf"), float("-inf")) for l in losses),
        "loss stayed finite every epoch (no NaN/inf)",
    )

    early = sum(losses[:3]) / len(losses[:3])
    late = sum(losses[-3:]) / len(losses[-3:])
    _check(
        late < early,
        f"loss improved over training (early avg={early:.4f} -> late avg={late:.4f})",
    )

    val_rows = [row for row in history if "val_MRR" in row]
    _check(len(val_rows) > 0, "validation metrics were computed at least once")

    # Checkpoint round-trip: a freshly constructed model loaded from the
    # saved checkpoint must reproduce identical parameters. Compared by
    # NAME (not position) so a mismatch reports exactly which parameter
    # differs, rather than a single opaque pass/fail.
    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    reloaded = build_model(
        config["model"], checkpoint["num_entities"], checkpoint["num_relations"]
    )
    reloaded.load_state_dict(checkpoint["model_state"])

    model_sd = model.to("cpu").state_dict()
    reloaded_sd = reloaded.state_dict()
    _check(
        set(model_sd.keys()) == set(reloaded_sd.keys()),
        f"checkpoint and reloaded model have the same parameter names "
        f"(only in trained model: {set(model_sd) - set(reloaded_sd)}, "
        f"only in reloaded: {set(reloaded_sd) - set(model_sd)})",
    )
    mismatches = [
        f"{name} (max abs diff={float((model_sd[name] - reloaded_sd[name]).abs().max()):.3g}, "
        f"shape={tuple(model_sd[name].shape)})"
        for name in model_sd
        if not torch.equal(model_sd[name], reloaded_sd[name])
    ]
    _check(
        not mismatches,
        "checkpoint reloads into a fresh model with identical parameters"
        + (f" -- MISMATCHED: {mismatches}" if mismatches else ""),
    )

    print("\nPhase 1 smoke test PASSED. Ready to train on the full FB15k-237 set.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
