"""Phase 1 training loop: KG Encoder Phase 1 + DistMult, no LM at all.

Config-driven so the same code runs both the toy-subset smoke test and the
eventual full FB15k-237 training run — see experiments/configs/phase1_*.yaml.
Encodes the whole graph once per optimizer step and reuses that encoding for
every positive/negative triple in the step (CLAUDE.md rule #7), rather than
recomputing it per triple.

Run directly as a script (`python -m training.train_kg_baseline <config.yaml>`),
or import `run(config)` from elsewhere (e.g. run_phase1_smoke_test.py).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import yaml

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.kg_only_baseline import KGOnlyBaseline
from preprocessing.dataset import load_dataset
from preprocessing.graph_builder import build_train_graph
from preprocessing.toy_subset import make_toy_subset
from training.losses import bce_loss
from training.negative_sampling import sample_negatives


def _edge_tensors(graph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t()
    edge_type = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    return edge_index, edge_type


def run(config: dict) -> dict:
    seed = config.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_cfg = config["dataset"]
    dataset = load_dataset(
        raw_dir=ds_cfg["raw_dir"],
        download_if_missing=ds_cfg.get("download_if_missing", True),
        use_synthetic_fallback=ds_cfg.get("use_synthetic_fallback", True),
    )

    if config.get("use_toy_subset", False):
        toy_cfg = config["toy_subset"]
        dataset = make_toy_subset(
            dataset, max_entities=toy_cfg["max_entities"], seed=toy_cfg["seed"]
        )

    print(
        f"Training on: entities={dataset.num_entities} "
        f"relations={dataset.num_relations} train={len(dataset.train)} "
        f"valid={len(dataset.valid)} test={len(dataset.test)}"
    )

    graph = build_train_graph(
        dataset.train, dataset.num_entities, dataset.num_relations
    )
    edge_index, edge_type = _edge_tensors(graph, device)

    model_cfg = config["model"]
    model = KGOnlyBaseline(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        dim=model_cfg["dim"],
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        num_bases=model_cfg.get("num_bases"),
    ).to(device)

    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    known_true_train: set[tuple[int, int, int]] = set(dataset.train)
    rng = random.Random(seed)

    # Filtering for evaluation uses every split combined (never used as
    # training signal — see evaluation/metrics.py docstring).
    hr_to_tails, rt_to_heads = build_filter_index(
        dataset.train + dataset.valid + dataset.test
    )

    train_triples_t = torch.tensor(dataset.train, dtype=torch.long, device=device)

    # When `save_best` is on, the checkpoint written at the end is the
    # best-validation-MRR epoch's parameters, not just whatever the last
    # epoch happened to produce — validation MRR is not guaranteed to
    # improve monotonically (confirmed by the Phase 1 toy smoke test run),
    # so for a real result this matters. Off by default so the existing toy
    # smoke test's "checkpoint == final in-memory model" assumption still
    # holds unless a config explicitly opts in.
    save_best = train_cfg.get("save_best", False)
    best_val_mrr = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    history: list[dict] = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(dataset.train))
        total_loss = 0.0
        steps = 0

        batch_size = train_cfg["batch_size"]
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            pos = train_triples_t[idx]

            neg = sample_negatives(
                pos.cpu(),
                dataset.num_entities,
                known_true_train,
                train_cfg["num_negatives"],
                rng,
            ).to(device)

            optimizer.zero_grad()
            entity_repr = model.encode(edge_index, edge_type)

            pos_scores = model.score_triples(entity_repr, pos)
            neg_flat = neg.view(-1, 3)
            neg_scores = model.score_triples(entity_repr, neg_flat).view(
                len(pos), train_cfg["num_negatives"]
            )

            loss = bce_loss(pos_scores, neg_scores)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip_norm", 1.0)
            )
            optimizer.step()

            total_loss += float(loss.detach())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        row = {"epoch": epoch, "loss": avg_loss}

        eval_every = train_cfg.get("eval_every", 1)
        if epoch % eval_every == 0 or epoch == train_cfg["epochs"]:
            model.eval()
            with torch.no_grad():
                eval_entity_repr = model.encode(edge_index, edge_type)
            if len(dataset.valid) > 0:
                val_metrics = evaluate_filtered(
                    model, eval_entity_repr, dataset.valid, hr_to_tails, rt_to_heads
                )
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                print(
                    f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                    f"val_MRR={val_metrics['MRR']:.4f} | "
                    f"val_Hits@10={val_metrics['Hits@10']:.4f}"
                )
                if save_best and val_metrics["MRR"] > best_val_mrr:
                    best_val_mrr = val_metrics["MRR"]
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
            else:
                print(
                    f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                    "(no validation triples in this subset)"
                )
        else:
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

        history.append(row)

    if save_best and best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restoring best checkpoint (val_MRR={best_val_mrr:.4f}) "
            "for saving — not necessarily the final epoch."
        )

    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "best_val_mrr": best_val_mrr if save_best else None,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    return {"history": history, "model": model, "dataset": dataset}


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "experiments/configs/phase1_toy.yaml"
    )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
