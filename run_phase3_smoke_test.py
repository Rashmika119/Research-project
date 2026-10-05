"""Run the trained R-GCN -> frozen RoBERTa integration gate on a toy graph.

Usage: python run_phase3_smoke_test.py [config.yaml] [--device cpu|cuda]
Requires real data, text and a compatible checkpoint with verified ID maps.
Does not train a replacement baseline or save/overwrite any model weights.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import torch
import yaml
from transformers import AutoTokenizer

from models.kg_lm_bridge import KGLMBridge
from preprocessing.dataset import load_dataset
from preprocessing.entity_text import align_entity_texts, download_entity_text_files
from preprocessing.relation_text import (
    align_relation_texts, download_relation_text_file, read_relation_text_mapping,
)
from training.phase3_gate import (
    check_bridge_paths, prepare_toy_model, read_checkpoint, sha256_file,
)


def run(config, device=None):
    if config["dataset"].get("use_synthetic_fallback", False):
        raise ValueError("Phase 3 checkpoint gate requires real data; synthetic fallback is disabled")
    if config["bridge"]["kg_dim"] != config["model"]["dim"]:
        raise ValueError("Bridge kg_dim must equal model.dim")
    seed = config.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    phase1 = config["phase1"]
    checkpoint, maps, provenance = read_checkpoint(
        phase1["checkpoint_path"], phase1.get("id_map_path")
    )
    dataset = load_dataset(**config["dataset"])
    model, toy, full_ids = prepare_toy_model(
        checkpoint, maps, dataset, config["model"],
        config["toy_subset"]["max_entities"], seed,
    )
    # Align using original string keys in the REMAPPED toy entity vocabulary.
    # Relation IDs remain those of the full dataset.
    text_dir = config["text"]["text_dir"]
    long_path, short_path = download_entity_text_files(text_dir)
    rel_path = download_relation_text_file(text_dir)
    entities = align_entity_texts(toy.entity2id, long_path, short_path)
    relations = align_relation_texts(toy.relation2id, read_relation_text_mapping(rel_path))
    if entities.missing or relations.missing:
        raise ValueError("Phase 3 real-text gate requires matched entity and relation descriptions")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3: R-GCN dim={config['model']['dim']}; device={device}")
    print(f"Toy graph: {toy.num_entities} entities, {len(toy.train)} training triples")
    tokenizer = AutoTokenizer.from_pretrained(config["bridge"]["model_name"])
    bridge = KGLMBridge(**config["bridge"])
    checks = check_bridge_paths(
        model, bridge, tokenizer, toy, entities.texts_by_id, relations.texts_by_id,
        batch_size=config["smoke_test"]["batch_size"],
        max_length=config["text"]["max_length"], device=device,
    )
    def git_output(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    status = git_output("status", "--porcelain")
    report = {
        "status": "passed", "gate": "phase3_checkpoint_real_text",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config": config, "device": str(device),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "torch-geometric", "transformers")},
        "checkpoint": {"path": phase1["checkpoint_path"], **provenance},
        "text_sha256": {str(p): sha256_file(p) for p in (long_path, short_path, rel_path)},
        "toy_entity2id": toy.entity2id, "relation2id": toy.relation2id,
        "toy_to_full_entity_ids": full_ids, "checks": checks,
        "note": "Toy graph recomputes representations from restored weights. No benchmark metrics or optimizer steps.",
    }
    # Unique files avoid mistaking an older pass report for the current attempt.
    output_dir = Path(config["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ("phase3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"Phase 3 integration gate PASSED. Report: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="experiments/configs/phase3_integration.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        run(config, args.device)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        parser.exit(1, f"Phase 3 integration gate FAILED: {exc}\n")


if __name__ == "__main__":
    main()
