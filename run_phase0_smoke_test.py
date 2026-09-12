"""Phase 0 smoke test — run this before starting Phase 1.

Usage (local machine with Python + deps installed):
    python run_phase0_smoke_test.py

In a Colab cell:
    !git clone <your-repo-url> repo && cd repo
    !pip install -r requirements.txt
    !python run_phase0_smoke_test.py

What this checks (mirrors the Phase 0 gate in CLAUDE.md):
  1. The dataset loads — real FB15k-237 if network access allows it,
     otherwise a synthetic fallback graph — and produces train/valid/test
     ID-triples with entity/relation maps that are internally consistent.
  2. No triple appears in both train and valid, or train and test.
  3. The training-only graph has exactly the expected number of edges
     (forward + inverse for every training triple, nothing extra).
  4. A toy subset can be derived from the real graph and is itself
     internally consistent (its own entity ids are freshly remapped, not
     leftover ids from the parent dataset).

Exits non-zero with a clear message if any check fails, so it can be used
as a CI gate as well as an interactive sanity check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from preprocessing.dataset import load_dataset
from preprocessing.graph_builder import assert_no_leakage, build_train_graph
from preprocessing.toy_subset import make_toy_subset

CONFIG_PATH = Path("experiments/configs/phase0_fb15k237.yaml")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"Phase 0 smoke test FAILED: {message}")
    print(f"  [ok] {message}")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    ds_cfg = config["dataset"]

    print(f"Loading dataset '{ds_cfg['name']}' from {ds_cfg['raw_dir']} ...")
    dataset = load_dataset(
        raw_dir=ds_cfg["raw_dir"],
        download_if_missing=ds_cfg["download_if_missing"],
        use_synthetic_fallback=ds_cfg["use_synthetic_fallback"],
    )

    print("\n--- Dataset summary ---")
    print(f"entities={dataset.num_entities} relations={dataset.num_relations}")
    print(
        f"train={len(dataset.train)} valid={len(dataset.valid)} "
        f"test={len(dataset.test)}"
    )

    # Real FB15k-237 is well-known to be 14,541 entities / 237 relations /
    # 272,115 train / 17,535 valid / 20,466 test. WN18's column order turned
    # out to be non-obvious and silently produced a nonsense relation count
    # that still passed every other check — so compare against the published
    # numbers here too, as an early warning rather than a hard failure (the
    # synthetic fallback graph, e.g. entities=40, is expected not to match).
    if dataset.num_entities > 1000:  # skip this check for the synthetic fallback
        expected = {
            "entities": 14541,
            "relations": 237,
            "train": 272115,
            "valid": 17535,
            "test": 20466,
        }
        actual = {
            "entities": dataset.num_entities,
            "relations": dataset.num_relations,
            "train": len(dataset.train),
            "valid": len(dataset.valid),
            "test": len(dataset.test),
        }
        mismatches = {
            k: (actual[k], expected[k])
            for k in expected
            if actual[k] != expected[k]
        }
        if mismatches:
            print(
                "\n  [WARNING] Loaded numbers don't match published FB15k-237 "
                "statistics — this is exactly the kind of mismatch that "
                "flagged WN18's column-order bug. Inspect a raw downloaded "
                "line (e.g. `head -n 3 " + ds_cfg["raw_dir"] + "/train.txt`) "
                "before trusting this data:"
            )
            for k, (got, exp) in mismatches.items():
                print(f"    {k}: got {got}, expected {exp}")
        else:
            print("  [ok] matches published FB15k-237 statistics exactly")

    print("\n--- Checks ---")
    _check(dataset.num_entities > 0, "at least one entity loaded")
    _check(dataset.num_relations > 0, "at least one relation loaded")
    _check(len(dataset.train) > 0, "train split is non-empty")

    max_train_id = max(max(h, t) for h, _, t in dataset.train)
    _check(
        max_train_id < dataset.num_entities,
        "max entity id in train triples is within entity2id range",
    )

    graph = build_train_graph(
        dataset.train, dataset.num_entities, dataset.num_relations
    )
    assert_no_leakage(graph, dataset.train, dataset.valid, dataset.test)
    _check(True, "train graph built from train triples only, no split overlap")
    _check(
        len(graph.edge_index) == len(dataset.train) * 2,
        "train graph has forward + inverse edges for every training triple",
    )

    toy_cfg = config["toy_subset"]
    toy = make_toy_subset(
        dataset, max_entities=toy_cfg["max_entities"], seed=toy_cfg["seed"]
    )
    print("\n--- Toy subset summary ---")
    print(f"entities={toy.num_entities} relations={toy.num_relations}")
    print(f"train={len(toy.train)} valid={len(toy.valid)} test={len(toy.test)}")

    _check(
        toy.num_entities <= toy_cfg["max_entities"],
        "toy subset respects max_entities",
    )
    _check(len(toy.train) > 0, "toy subset has at least one training triple")
    toy_max_id = max(max(h, t) for h, _, t in toy.train)
    _check(
        toy_max_id < toy.num_entities,
        "toy subset entity ids are self-consistent (no stale original ids)",
    )

    print("\nPhase 0 smoke test PASSED. Ready to start Phase 1 (KG-only baseline).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface a clean failure for CI / Colab logs
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
