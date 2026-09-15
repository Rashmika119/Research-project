"""Check long-description coverage for FB15k-237 entities."""

from preprocessing.dataset import load_dataset
from preprocessing.entity_text import read_text_mapping


DATA_DIR = "data/raw/fb15k237"

LONG_TEXT_PATH = (
    "data/text/fb15k237/FB15k_mid2description.txt"
)

SHORT_TEXT_PATH = (
    "data/text/fb15k237/entity2text.txt"
)


def main() -> None:
    print(
        "=== FB15k-237 Long Description Alignment Check ===\n"
    )

    dataset = load_dataset(
        raw_dir=DATA_DIR,
        download_if_missing=True,
        use_synthetic_fallback=False,
    )

    long_text = read_text_mapping(
        LONG_TEXT_PATH
    )

    short_text = read_text_mapping(
        SHORT_TEXT_PATH
    )

    long_matched = 0
    short_fallback = 0
    missing = 0

    long_missing_examples: list[str] = []

    for entity_key in dataset.entity2id:

        long_description = (
            long_text.get(entity_key, "").strip()
        )

        short_description = (
            short_text.get(entity_key, "").strip()
        )

        if long_description:
            long_matched += 1

        elif short_description:
            short_fallback += 1

            if len(long_missing_examples) < 10:
                long_missing_examples.append(
                    entity_key
                )

        else:
            missing += 1

    total = dataset.num_entities

    final_matched = (
        long_matched
        + short_fallback
    )

    long_coverage = (
        long_matched / total * 100
    )

    final_coverage = (
        final_matched / total * 100
    )

    print(f"Total KG entities     : {total}")
    print(f"Long descriptions     : {long_matched}")
    print(f"Short-text fallbacks  : {short_fallback}")
    print(f"Still missing         : {missing}")
    print(
        f"Long-text coverage    : "
        f"{long_coverage:.2f}%"
    )
    print(
        f"Final coverage        : "
        f"{final_coverage:.2f}%"
    )

    if long_missing_examples:
        print(
            "\nExamples requiring short-text fallback:"
        )

        for entity_key in long_missing_examples:
            print(f"  {entity_key}")

    assert total == 14541

    assert final_matched == total, (
        "Long + short text does not cover "
        "all KG entities"
    )

    print(
        "\nLong-description alignment check PASSED."
    )


if __name__ == "__main__":
    main()