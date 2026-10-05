"""Checkpoint-backed Phase 3 wiring checks; no optimizer or benchmark training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from preprocessing.graph_builder import assert_no_leakage, build_train_graph
from preprocessing.toy_subset import make_toy_subset
from training.model_factory import build_model


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: str | Path, id_map_path: str | Path | None = None):
    """Fail before downloads; legacy ID maps must be bound to these file bytes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 3 requires the trained Phase 1 checkpoint: {path}. "
            "Recover it or train a new recorded baseline; random fallback is disabled."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    digest = sha256_file(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a Phase 1 checkpoint dictionary")
    required = {"model_state", "config", "num_entities", "num_relations"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint missing fields: {sorted(required - checkpoint.keys())}")
    maps = {key: checkpoint.get(key) for key in ("entity2id", "relation2id")}
    source = "checkpoint"
    if any(value is None for value in maps.values()):
        if not id_map_path:
            raise ValueError(
                "Legacy checkpoint has no complete ID maps. Set phase1.id_map_path "
                "to a verified JSON sidecar containing checkpoint_sha256, entity2id "
                "and relation2id from the original run. Counts alone cannot prove identity."
            )
        sidecar = json.loads(Path(id_map_path).read_text(encoding="utf-8"))
        if sidecar.get("checkpoint_sha256") != digest:
            raise ValueError("ID-map sidecar checkpoint_sha256 does not match checkpoint")
        for key in maps:
            if maps[key] is not None and maps[key] != sidecar.get(key):
                raise ValueError(f"Sidecar contradicts checkpoint {key}")
        maps = {key: sidecar.get(key) for key in maps}
        source = str(id_map_path)
    for key, mapping in maps.items():
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing or invalid {key}")
        if (not all(isinstance(k, str) and type(v) is int for k, v in mapping.items())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError(f"{key} must map unique names to contiguous integer IDs")
    return checkpoint, maps, {"sha256": digest, "id_map_source": source}


def _model_settings(config):
    return {
        "encoder_type": config.get("encoder_type", "rgcn"),
        "dim": config["dim"],
        "num_layers": config.get("num_layers", 2),
        "dropout": config.get("dropout", 0.2),
        "num_bases": config.get("num_bases"),
    }


def prepare_toy_model(checkpoint, maps, dataset, model_config, max_entities, seed):
    """Strictly restore full weights, then gather entity rows by their KG keys.

    The toy graph keeps full relation IDs, including inverse-edge offsets.
    Encoder outputs are recomputed on the toy training graph, so they are not
    claimed to equal full-graph representations.
    """
    settings = _model_settings(model_config)
    if settings["encoder_type"] != "rgcn":
        raise ValueError("This Phase 3 gate is configured for the selected R-GCN encoder")
    if _model_settings(checkpoint["config"]["model"]) != settings:
        raise ValueError("Checkpoint encoder settings do not match the integration config")
    for key, count in (("entity2id", "num_entities"), ("relation2id", "num_relations")):
        if checkpoint[count] != getattr(dataset, count):
            raise ValueError(f"Checkpoint {count} does not match dataset")
        if maps[key] != getattr(dataset, key):
            raise ValueError(f"Checkpoint {key} does not match dataset identity/order")
    if max_entities < 2:
        raise ValueError("toy_subset.max_entities must be at least 2")
    full = build_model(settings, dataset.num_entities, dataset.num_relations)
    full.load_state_dict(checkpoint["model_state"], strict=True)
    if not all(torch.isfinite(t).all() for t in full.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model tensors")

    toy = make_toy_subset(dataset, max_entities=max_entities, seed=seed)
    if toy.relation2id != dataset.relation2id:
        raise ValueError("Toy relation IDs changed; encoder relation weights would be misaligned")
    keys = sorted(toy.entity2id, key=toy.entity2id.get)
    full_ids = [dataset.entity2id[key] for key in keys]
    model = build_model(settings, toy.num_entities, toy.num_relations)
    state = dict(full.state_dict())
    state["entity_emb.weight"] = full.entity_emb.weight.detach()[full_ids].clone()
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.entity_emb.weight, full.entity_emb.weight[full_ids]):
        raise AssertionError("Entity-row transfer changed checkpoint values")
    return model, toy, full_ids


def gradient_norm(module, label):
    """All trainable parameters must have finite grads; group norm must be > 0."""
    total = 0.0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise AssertionError(f"{label}.{name}: missing or non-finite gradient")
        total += float(param.grad.detach().double().square().sum())
    if not total > 0:
        raise AssertionError(f"{label}: zero gradient")
    return total ** 0.5


def _check_frozen(bridge):
    if bridge.lm.lm.training:
        raise AssertionError("RoBERTa must remain in eval mode")
    if any(p.requires_grad or p.grad is not None for p in bridge.lm.parameters()):
        raise AssertionError("RoBERTa must be frozen with no parameter gradients")


def check_bridge_paths(model, bridge, tokenizer, toy, entity_texts, relation_texts,
                       batch_size=4, max_length=64, device="cpu"):
    """Check entity and relation branches independently so neither masks a break."""
    if batch_size < 1 or max_length < 2:
        raise ValueError("Positive batch size and max_length >= 2 are required")
    if len(entity_texts) != toy.num_entities or len(relation_texts) != toy.num_relations:
        raise ValueError("Text arrays must use the toy entity/full relation ID ranges")
    if bridge.kg_dim != model.entity_emb.embedding_dim:
        raise ValueError("Bridge and encoder KG dimensions differ")
    graph = build_train_graph(toy.train, toy.num_entities, toy.num_relations)
    assert_no_leakage(graph, toy.train, toy.valid, toy.test)
    if not toy.train:
        raise ValueError("Toy graph has no training triples")
    model.to(device).train()
    bridge.to(device).train()
    _check_frozen(bridge)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t()
    edge_type = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    entity_ids = sorted({e for h, _, t in toy.train for e in (h, t)})[:batch_size]
    relation_ids = sorted({r for _, r, _ in toy.train})[:batch_size]
    report = {}
    for kind, ids, texts in (("entity", entity_ids, entity_texts),
                             ("relation", relation_ids, relation_texts)):
        descriptions = [texts[i] for i in ids]
        if any(not text.strip() for text in descriptions):
            raise ValueError(f"Missing selected {kind} description")
        tokens = tokenizer(descriptions, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        model.zero_grad(set_to_none=True)
        bridge.zero_grad(set_to_none=True)
        if kind == "entity":
            # One graph encode, then gather real structure-aware outputs.
            vectors = model.encode(edge_index, edge_type)[ids]
        else:
            index = torch.tensor(ids, dtype=torch.long, device=device)
            vectors = model.scorer.relation_emb(index)
        vectors.retain_grad()
        output = bridge(vectors, tokens["input_ids"].to(device),
                        tokens["attention_mask"].to(device))
        if output.shape != (len(ids), bridge.kg_dim) or not torch.isfinite(output).all():
            raise AssertionError(f"{kind}: invalid bridge output")
        # Mechanical differentiability diagnostic, not the link-prediction loss.
        loss = output.square().mean()
        loss.backward()
        if (vectors.grad is None or not torch.isfinite(vectors.grad).all()
                or not (vectors.grad.abs().sum(dim=1) > 0).all()):
            raise AssertionError(f"{kind}: missing, non-finite or zero input gradient")
        norms = {
            "kg_to_lm": gradient_norm(bridge.kg_to_lm, f"{kind}.kg_to_lm"),
            "lm_to_kg": gradient_norm(bridge.lm_to_kg, f"{kind}.lm_to_kg"),
        }
        if kind == "entity":
            norms["encoder"] = gradient_norm(model.encoder, "encoder")
            norms["entity_embeddings"] = gradient_norm(model.entity_emb, "entity_embeddings")
        else:
            norms["relation_embeddings"] = gradient_norm(model.scorer.relation_emb,
                                                          "relation_embeddings")
            if not (model.scorer.relation_emb.weight.grad[ids].abs().sum(dim=1) > 0).all():
                raise AssertionError("Selected relation embedding row has zero gradient")
        _check_frozen(bridge)
        report[kind] = {"ids": ids, "output_shape": list(output.shape),
                        "diagnostic_loss": float(loss.detach()), "gradient_norms": norms}
    return report
