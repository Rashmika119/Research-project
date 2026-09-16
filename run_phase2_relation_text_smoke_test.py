"""Phase 2 smoke test using real FB15k-237 relation text.

This test uses dummy 32-dimensional relation vectors because the
learned relation embeddings from the KG scoring component will be
connected during later integration.

What is real in this test:
    - FB15k-237 relation IDs
    - relation-to-text alignment
    - real relation text
    - RoBERTa tokenizer input
    - frozen RoBERTa KG-LM bridge
"""

from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoTokenizer

from models.kg_lm_bridge import KGLMBridge
from preprocessing.dataset import load_dataset
from preprocessing.relation_text import (
    align_relation_texts,
    download_relation_text_file,
    read_relation_text_mapping,
)


SEED = 0

DATA_DIR = "data/raw/fb15k237"
TEXT_DIR = "data/text/fb15k237"

MODEL_NAME = "roberta-base"

KG_DIM = 32
BATCH_SIZE = 4
MAX_LENGTH = 64


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    print(
        "=== Phase 2 Real FB15k-237 Relation Text Smoke Test ===\n"
    )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------------
    # 1. Load the real FB15k-237 KG.
    # --------------------------------------------------------------
    dataset = load_dataset(
        raw_dir=DATA_DIR,
        download_if_missing=True,
        use_synthetic_fallback=False,
    )

    print(f"KG relations: {dataset.num_relations}")

    assert dataset.num_relations == 237, (
        f"Expected 237 relations, got {dataset.num_relations}"
    )

    # --------------------------------------------------------------
    # 2. Load real relation text.
    # --------------------------------------------------------------
    relation_text_path = download_relation_text_file(
        TEXT_DIR
    )

    print(
        f"Relation text file: {relation_text_path}"
    )

    relation_text_mapping = read_relation_text_mapping(
        relation_text_path
    )

    # --------------------------------------------------------------
    # 3. Align relation text to existing relation2id.
    # --------------------------------------------------------------
    alignment = align_relation_texts(
        relation2id=dataset.relation2id,
        relation_text_mapping=relation_text_mapping,
    )

    coverage = (
        alignment.matched
        / dataset.num_relations
        * 100.0
    )

    print(f"Text coverage: {coverage:.2f}%")
    print(f"Matched text : {alignment.matched}")
    print(f"Missing text : {alignment.missing}")

    assert alignment.missing == 0, (
        "Some FB15k-237 relations have no usable text"
    )

    assert alignment.matched == dataset.num_relations, (
        "Relation text coverage does not match relation count"
    )

    # --------------------------------------------------------------
    # 4. Select real relation IDs from training triples.
    # --------------------------------------------------------------
    selected_ids: list[int] = []

    for _, relation_id, _ in dataset.train:
        if relation_id not in selected_ids:
            selected_ids.append(relation_id)

        if len(selected_ids) == BATCH_SIZE:
            break

    assert len(selected_ids) == BATCH_SIZE

    id2relation = {
        relation_id: relation_key
        for relation_key, relation_id
        in dataset.relation2id.items()
    }

    descriptions = [
        alignment.texts_by_id[relation_id]
        for relation_id in selected_ids
    ]

    # --------------------------------------------------------------
    # 5. Validate relation text.
    # --------------------------------------------------------------
    for relation_id, description in zip(
        selected_ids,
        descriptions,
    ):
        assert description.strip(), (
            f"Empty text for relation ID {relation_id}"
        )

    print(
        "\n--- Real relation/text examples ---"
    )

    for relation_id, description in zip(
        selected_ids,
        descriptions,
    ):
        relation_key = id2relation[relation_id]

        print(
            f"ID {relation_id} | {relation_key}"
        )
        print(
            f"  {description}"
        )

    # --------------------------------------------------------------
    # 6. Tokenize real relation descriptions.
    # --------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoded = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(
        device
    )

    print(
        f"\nTokenized shape: "
        f"{tuple(input_ids.shape)}"
    )

    # --------------------------------------------------------------
    # 7. Dummy relation vectors.
    #
    # Later integration will replace these with learned
    # 32-D relation embeddings.
    # --------------------------------------------------------------
    relation_vectors = torch.randn(
        BATCH_SIZE,
        KG_DIM,
        device=device,
        requires_grad=True,
    )

    # --------------------------------------------------------------
    # 8. Run the SAME KG-LM bridge used for entities.
    # --------------------------------------------------------------
    model = KGLMBridge(
        kg_dim=KG_DIM,
        model_name=MODEL_NAME,
        projection_dropout=0.1,
    ).to(device)

    model.train()

    output = model(
        kg_vectors=relation_vectors,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert output.shape == (
        BATCH_SIZE,
        KG_DIM,
    ), (
        f"Unexpected output shape: "
        f"{tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "Relation semantic output contains NaN or inf"
    )

    # --------------------------------------------------------------
    # 9. Gradient-flow check.
    # --------------------------------------------------------------
    loss = output.pow(2).mean()
    loss.backward()

    assert relation_vectors.grad is not None, (
        "Gradient did not reach relation vectors"
    )

    assert any(
        parameter.grad is not None
        for parameter
        in model.kg_to_lm.parameters()
    ), (
        "KG -> LM projection received no gradient"
    )

    assert any(
        parameter.grad is not None
        for parameter
        in model.lm_to_kg.parameters()
    ), (
        "LM -> KG projection received no gradient"
    )

    assert not any(
        parameter.grad is not None
        for parameter
        in model.lm.lm.parameters()
    ), (
        "Frozen RoBERTa unexpectedly received "
        "parameter gradients"
    )

    # --------------------------------------------------------------
    # 10. Final checks.
    # --------------------------------------------------------------
    print(
        "\n--- Relation bridge checks ---"
    )

    print(
        f"[ok] relation input shape  = "
        f"{tuple(relation_vectors.shape)}"
    )

    print(
        f"[ok] relation output shape = "
        f"{tuple(output.shape)}"
    )

    print(
        "[ok] real FB15k-237 relation text tokenized"
    )

    print(
        "[ok] gradients reach relation vectors"
    )

    print(
        "[ok] gradients reach trainable projections"
    )

    print(
        "[ok] frozen RoBERTa receives no "
        "parameter gradients"
    )

    print(
        "\nPhase 2 RELATION-TEXT smoke test PASSED."
    )


if __name__ == "__main__":
    main()