"""Check FB15k-237 entity-text alignment against the existing KG entity IDs."""

from preprocessing.dataset import load_dataset
from preprocessing.entity_text import read_text_mapping


DATA_DIR = "data/raw/fb15k237"
TEXT_PATH = "data/text/fb15k237/entity2text.txt"


def main() -> None:
    print("=== FB15k-237 Entity Text Alignment Check ===\n")

    # Load the real FB15k-237 dataset.
    dataset = load_dataset(
        raw_dir=DATA_DIR,
        download_if_missing=True,
        use_synthetic_fallback=False,
    )

    # Load entity text keyed by the original Freebase entity key.
    text_mapping = read_text_mapping(TEXT_PATH)

    total_entities = dataset.num_entities

    matched = 0
    missing_entities: list[str] = []

    for entity_key in dataset.entity2id:
        text = text_mapping.get(entity_key, "").strip()

        if text:
            matched += 1
        else:
            missing_entities.append(entity_key)

    missing = total_entities - matched
    coverage = (
        matched / total_entities * 100
        if total_entities
        else 0.0
    )

    print(f"Total KG entities : {total_entities}")
    print(f"Matched text      : {matched}")
    print(f"Missing text      : {missing}")
    print(f"Coverage          : {coverage:.2f}%")

    # Show only a few missing examples, if any.
    if missing_entities:
        print("\nFirst missing entity keys:")
        for entity_key in missing_entities[:10]:
            print(f"  {entity_key}")

    # Basic safety checks.
    assert total_entities == 14541, (
        f"Expected 14,541 FB15k-237 entities, got {total_entities}"
    )

    assert matched > 0, (
        "No entity texts matched the KG entity keys"
    )

    print("\nEntity text alignment check PASSED.")


if __name__ == "__main__":
    main()