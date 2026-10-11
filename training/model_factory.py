"""Construct the shared RGAT + ComplEx structural model from saved settings."""

from __future__ import annotations

import torch.nn as nn

from models.kg_only_baseline_rgat import KGOnlyBaselineRGAT


def build_model(
    model_cfg: dict, num_entities: int, num_relations: int
) -> nn.Module:
    """`model_cfg` is a config's `model:` block (dim, num_layers, ...)."""
    encoder_type = model_cfg.get("encoder_type", "rgat")
    common = dict(
        num_entities=num_entities,
        num_relations=num_relations,
        dim=model_cfg["dim"],
        scorer_type=model_cfg.get("scorer_type", "complex"),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        num_bases=model_cfg.get("num_bases"),
    )
    if encoder_type == "rgat":
        return KGOnlyBaselineRGAT(heads=model_cfg.get("heads", 2), **common)
    raise ValueError(f"Unknown model.encoder_type: {encoder_type!r}")
