from __future__ import annotations

from preprocessing.dataset import load_fb15k237
from preprocessing.relation_text import (
    align_relation_texts,
    download_relation_text_file,
    read_relation_text_mapping,
)


RAW_DIR = "data/raw/fb15k237"
TEXT_DIR = "data/text/fb15k237"


def main() -> None:
    print("=== FB15k-237 Relation Text Alignment Check ===")

    dataset = load_fb15k237(
        RAW_DIR,
        download_if_missing=True,
    )

    print(f"KG relations: {dataset.num_relations}")

    relation_text_path = download_relation_text_file(TEXT_DIR)

    print(f"Relation text file: {relation_text_path}")

    relation_text_mapping = read_relation_text_mapping(
        relation_text_path
    )

    print(
        f"Relation text records: "
        f"{len(relation_text_mapping)}"
    )

    alignment = align_relation_texts(
        dataset.relation2id,
        relation_text_mapping,
    )

    total_relations = dataset.num_relations

    coverage = (
        alignment.matched / total_relations * 100.0
        if total_relations
        else 0.0
    )

    print()
    print("--- Alignment summary ---")
    print(f"Total relations : {total_relations}")
    print(f"Matched text    : {alignment.matched}")
    print(f"Missing text    : {alignment.missing}")
    print(f"Coverage        : {coverage:.2f}%")

    assert total_relations == 237, (
        f"Expected 237 FB15k-237 relations, "
        f"got {total_relations}"
    )

    assert len(alignment.texts_by_id) == total_relations

    assert all(
        text.strip()
        for text in alignment.texts_by_id
    ), "Every relation ID must have non-empty text."

    print()
    print("--- Example relation texts ---")

    id2relation = {
        relation_id: relation_key
        for relation_key, relation_id
        in dataset.relation2id.items()
    }

    for relation_id in range(min(5, total_relations)):
        print(
            f"ID {relation_id} | "
            f"{id2relation[relation_id]}"
        )
        print(
            f"  {alignment.texts_by_id[relation_id]}"
        )

    print()
    print("Relation text alignment check PASSED.")


if __name__ == "__main__":
    main()