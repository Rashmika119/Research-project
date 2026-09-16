"""Phase 2 smoke test using real FB15k-237 entity descriptions.

This still uses dummy KG vectors because the trained RGAT output is a
Phase 1 dependency that will be connected later.

What is real in this test:
    - FB15k-237 entity IDs
    - entity-to-text alignment
    - automatically downloaded entity text resources
    - long descriptions
    - short-text fallback
    - tokenizer input
    - frozen BERT bridge
"""

from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoTokenizer

from preprocessing.dataset import load_dataset
from preprocessing.entity_text import (
    align_entity_texts,
    download_entity_text_files,
)
from models.kg_lm_bridge import KGLMBridge


SEED = 0

DATA_DIR = "data/raw/fb15k237"
TEXT_DIR = "data/text/fb15k237"

MODEL_NAME = "bert-base-uncased"

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
        "=== Phase 2 Real FB15k-237 Text Smoke Test ===\n"
    )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load the real FB15k-237 KG.
    #
    # The existing entity2id mapping created here is the ONLY entity
    # vocabulary used by this test.
    # ------------------------------------------------------------------
    dataset = load_dataset(
        raw_dir=DATA_DIR,
        download_if_missing=True,
        use_synthetic_fallback=False,
    )

    print(
        f"KG entities: {dataset.num_entities}"
    )

    # ------------------------------------------------------------------
    # 2. Download entity-text resources if missing.
    #
    # This makes the test reproducible in fresh environments such as
    # Google Colab without manually downloading text files first.
    # ------------------------------------------------------------------
    (
        long_text_path,
        short_text_path,
    ) = download_entity_text_files(
        TEXT_DIR
    )

    print(
        f"Long text file : {long_text_path}"
    )

    print(
        f"Short text file: {short_text_path}"
    )

    # ------------------------------------------------------------------
    # 3. Align real descriptions to the EXACT KG entity IDs.
    #
    # Priority:
    #   long description
    #   -> short text fallback
    #   -> missing
    # ------------------------------------------------------------------
    alignment = align_entity_texts(
        entity2id=dataset.entity2id,
        long_text_path=long_text_path,
        short_text_path=short_text_path,
    )

    print(
        f"Text coverage: "
        f"{alignment.coverage * 100:.2f}%"
    )

    print(
        f"Matched text : {alignment.matched}"
    )

    print(
        f"Missing text : {alignment.missing}"
    )

    assert alignment.missing == 0, (
        "Some FB15k-237 entities still have no usable text"
    )

    assert alignment.matched == dataset.num_entities, (
        "Text coverage does not match the KG entity count"
    )

    # ------------------------------------------------------------------
    # 4. Select a few REAL entity IDs appearing in training triples.
    # ------------------------------------------------------------------
    selected_ids: list[int] = []

    for h, _, t in dataset.train:
        for entity_id in (h, t):

            if entity_id not in selected_ids:
                selected_ids.append(
                    entity_id
                )

            if len(selected_ids) == BATCH_SIZE:
                break

        if len(selected_ids) == BATCH_SIZE:
            break

    assert len(selected_ids) == BATCH_SIZE

    # Reverse mapping used only for readable diagnostics.
    id2entity = {
        entity_id: entity_key
        for entity_key, entity_id
        in dataset.entity2id.items()
    }

    descriptions = [
        alignment.texts_by_id[entity_id]
        for entity_id in selected_ids
    ]

    # ------------------------------------------------------------------
    # 5. Verify cleaned descriptions.
    # ------------------------------------------------------------------
    for entity_id, description in zip(
        selected_ids,
        descriptions,
    ):
        assert description.strip(), (
            f"Empty description for entity ID {entity_id}"
        )

        assert not description.endswith("@en"), (
            "Language suffix @en was not cleaned"
        )

        assert "\\n" not in description, (
            "Literal newline escape was not cleaned"
        )

    print(
        "\n--- Real entity/text examples ---"
    )

    for entity_id, description in zip(
        selected_ids,
        descriptions,
    ):
        entity_key = id2entity[
            entity_id
        ]

        preview = (
            description[:120] + "..."
            if len(description) > 120
            else description
        )

        print(
            f"ID {entity_id} | {entity_key}"
        )

        print(
            f"  {preview}"
        )

    # ------------------------------------------------------------------
    # 6. Tokenize REAL entity descriptions.
    # ------------------------------------------------------------------
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

    input_ids = (
        encoded["input_ids"]
        .to(device)
    )

    attention_mask = (
        encoded["attention_mask"]
        .to(device)
    )

    print(
        f"\nTokenized shape: "
        f"{tuple(input_ids.shape)}"
    )

    # ------------------------------------------------------------------
    # 7. Dummy KG vectors imitate future Phase 1 RGAT outputs.
    #
    # Later Phase 3 integration will replace these vectors with:
    #
    #   real_rgAT_entity_repr[selected_ids]
    # ------------------------------------------------------------------
    kg_vectors = torch.randn(
        BATCH_SIZE,
        KG_DIM,
        device=device,
        requires_grad=True,
    )

    # ------------------------------------------------------------------
    # 8. Run the complete Phase 2 bridge.
    # ------------------------------------------------------------------
    model = KGLMBridge(
        kg_dim=KG_DIM,
        model_name=MODEL_NAME,
        projection_dropout=0.1,
    ).to(device)

    model.train()

    output = model(
        kg_vectors=kg_vectors,
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

    assert torch.isfinite(
        output
    ).all(), (
        "Phase 2 output contains NaN or inf"
    )

    # ------------------------------------------------------------------
    # 9. Gradient-flow test.
    # ------------------------------------------------------------------
    loss = output.pow(2).mean()

    loss.backward()

    assert kg_vectors.grad is not None, (
        "Gradient did not reach the KG input"
    )

    assert any(
        p.grad is not None
        for p in model.kg_to_lm.parameters()
    ), (
        "KG -> LM projection received no gradient"
    )

    assert any(
        p.grad is not None
        for p in model.lm_to_kg.parameters()
    ), (
        "LM -> KG projection received no gradient"
    )

    assert not any(
        p.grad is not None
        for p in model.lm.lm.parameters()
    ), (
        "Frozen BERT unexpectedly received "
        "parameter gradients"
    )

    # ------------------------------------------------------------------
    # 10. Final checks.
    # ------------------------------------------------------------------
    print(
        "\n--- Bridge checks ---"
    )

    print(
        f"[ok] KG input shape  = "
        f"{tuple(kg_vectors.shape)}"
    )

    print(
        f"[ok] KG output shape = "
        f"{tuple(output.shape)}"
    )

    print(
        "[ok] real FB15k-237 descriptions tokenized"
    )

    print(
        "[ok] cleaned descriptions contain no "
        "raw @en / \\\\n markers"
    )

    print(
        "[ok] gradients reach trainable projections"
    )

    print(
        "[ok] frozen BERT receives no parameter gradients"
    )

    print(
        "\nPhase 2 REAL-TEXT smoke test PASSED."
    )


if __name__ == "__main__":
    main()