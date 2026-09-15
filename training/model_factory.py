"""Builds a Phase 1 model from its config's `model` block.

One place that maps `model.encoder_type` ("rgcn" or "rgat") to the right
model class, so `training/train_kg_baseline.py` and the Phase 1 smoke-test
scripts can't drift out of sync on how a config gets turned into a model.
"""

from __future__ import annotations

import torch.nn as nn

from models.kg_only_baseline import KGOnlyBaseline
from models.kg_only_baseline_rgat import KGOnlyBaselineRGAT


def build_model(
    model_cfg: dict, num_entities: int, num_relations: int
) -> nn.Module:
    """`model_cfg` is a config's `model:` block (dim, num_layers, ...)."""
    encoder_type = model_cfg.get("encoder_type", "rgcn")
    common = dict(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=model_cfg["dim"],
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        num_bases=model_cfg.get("num_bases"),
    )
    if encoder_type == "rgcn":
        return KGOnlyBaseline(**common)
    if encoder_type == "rgat":
        return KGOnlyBaselineRGAT(heads=model_cfg.get("heads", 2), **common)
    raise ValueError(f"Unknown model.encoder_type: {encoder_type!r}")
