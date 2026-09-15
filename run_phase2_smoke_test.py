"""Phase 2 standalone smoke test: KG -> Frozen LM -> KG.

This test does NOT require a trained Phase 1 RGAT checkpoint.

It uses dummy KG vectors to verify that:

1. KG vectors have the expected dimension.
2. KG -> LM projection works.
3. The frozen LM processes the soft prompt + entity text.
4. LM -> KG projection returns to KG space.
5. Output shapes are correct.
6. Outputs remain finite (no NaN/inf).
7. Gradients flow through both projection modules.
8. Gradients can flow back to the Phase 1 input.
9. Frozen LM parameters receive no gradients.
10. Frozen LM remains in evaluation mode.
11. Fixed-input evaluation is deterministic.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from models.kg_lm_bridge import KGLMBridge


CONFIG_PATH = Path("experiments/configs/phase2_lm_toy.yaml")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    print("=== Phase 2: KG -> LM -> KG smoke test ===\n")

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])

    kg_dim = int(config["model"]["kg_dim"])
    lm_name = str(config["model"]["lm_name"])
    projection_dropout = float(
        config["model"]["projection_dropout"]
    )

    max_length = int(config["text"]["max_length"])
    batch_size = int(config["smoke_test"]["batch_size"])

    set_seed(seed)

    # ------------------------------------------------------------------
    # 2. Device
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"LM: {lm_name}")
    print(f"KG dimension: {kg_dim}")
    print(f"Batch size: {batch_size}\n")

    # ------------------------------------------------------------------
    # 3. Tokenizer + model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        lm_name
    )

    model = KGLMBridge(
        kg_dim=kg_dim,
        model_name=lm_name,
        projection_dropout=projection_dropout,
    ).to(device)

    # ------------------------------------------------------------------
    # 4. Dummy entity descriptions
    # ------------------------------------------------------------------
    descriptions = [
        "Albert Einstein was a theoretical physicist.",
        "Paris is the capital city of France.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Python is a high-level programming language.",
    ]

    if batch_size > len(descriptions):
        # Repeat descriptions only for smoke testing if a larger toy
        # batch size is configured later.
        repeats = (
            batch_size + len(descriptions) - 1
        ) // len(descriptions)

        descriptions = (
            descriptions * repeats
        )[:batch_size]

    else:
        descriptions = descriptions[:batch_size]

    encoded = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # 5. Dummy KG vectors
    #
    # These imitate future Phase 1 RGAT outputs.
    # ------------------------------------------------------------------
    kg_vectors = torch.randn(
        batch_size,
        kg_dim,
        device=device,
        requires_grad=True,
    )

    # ------------------------------------------------------------------
    # 6. Forward-path test
    # ------------------------------------------------------------------
    model.train()

    output = model(
        kg_vectors=kg_vectors,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    expected_shape = (
        batch_size,
        kg_dim,
    )

    assert tuple(output.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, "
        f"got {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "Phase 2 output contains NaN or inf"
    )

    print("--- Forward checks ---")
    print(
        f"[ok] input shape  = "
        f"{tuple(kg_vectors.shape)}"
    )
    print(
        f"[ok] output shape = "
        f"{tuple(output.shape)}"
    )
    print(
        "[ok] output contains only finite values"
    )

    # ------------------------------------------------------------------
    # 7. Frozen-LM checks
    # ------------------------------------------------------------------
    lm_parameters = list(
        model.lm.lm.parameters()
    )

    assert all(
        not p.requires_grad
        for p in lm_parameters
    ), "At least one LM parameter is trainable"

    assert model.lm.lm.training is False, (
        "Frozen LM must remain in eval mode"
    )

    print("\n--- Frozen LM checks ---")
    print(
        "[ok] all LM parameters are frozen"
    )
    print(
        "[ok] frozen LM remains in eval mode"
    )

    # ------------------------------------------------------------------
    # 8. Backward / gradient-flow test
    # ------------------------------------------------------------------
    loss = output.pow(2).mean()

    loss.backward()

    input_has_grad = (
        kg_vectors.grad is not None
        and torch.isfinite(
            kg_vectors.grad
        ).all()
    )

    kg_to_lm_has_grad = any(
        p.grad is not None
        for p in model.kg_to_lm.parameters()
    )

    lm_to_kg_has_grad = any(
        p.grad is not None
        for p in model.lm_to_kg.parameters()
    )

    lm_has_grad = any(
        p.grad is not None
        for p in lm_parameters
    )

    assert input_has_grad, (
        "Gradient did not reach the KG input"
    )

    assert kg_to_lm_has_grad, (
        "KG -> LM projection received no gradient"
    )

    assert lm_to_kg_has_grad, (
        "LM -> KG projection received no gradient"
    )

    assert not lm_has_grad, (
        "Frozen LM unexpectedly received gradients"
    )

    print("\n--- Gradient checks ---")
    print(
        "[ok] gradient reaches KG input"
    )
    print(
        "[ok] KG -> LM projection receives gradients"
    )
    print(
        "[ok] LM -> KG projection receives gradients"
    )
    print(
        "[ok] frozen LM receives no parameter gradients"
    )

    # ------------------------------------------------------------------
    # 9. Determinism check
    #
    # Projection dropout is disabled in eval mode.
    # With the same fixed input, two forwards should match.
    # ------------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        eval_output_1 = model(
            kg_vectors=kg_vectors.detach(),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        eval_output_2 = model(
            kg_vectors=kg_vectors.detach(),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    deterministic = torch.allclose(
        eval_output_1,
        eval_output_2,
        atol=1e-6,
        rtol=1e-5,
    )

    assert deterministic, (
        "Fixed-input evaluation is not deterministic"
    )

    print("\n--- Reproducibility check ---")
    print(
        "[ok] fixed-input evaluation is deterministic"
    )

    # ------------------------------------------------------------------
    # 10. Final result
    # ------------------------------------------------------------------
    print(
        "\nPhase 2 smoke test PASSED. "
        "KG -> Frozen LM -> KG bridge is ready "
        "for later Phase 1 RGAT integration."
    )


if __name__ == "__main__":
    main()