from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


RELATION_TEXT_URL = (
    "https://raw.githubusercontent.com/"
    "yao8839836/kg-bert/master/data/FB15k-237/relation2text.txt"
)


@dataclass
class RelationTextAlignment:
    texts_by_id: list[str]
    matched: int
    missing: int


def download_relation_text_file(
    text_dir: str | Path = "data/text/fb15k237",
) -> Path:
    """
    Download the FB15k-237 relation-to-text mapping if it is not
    already available locally.
    """
    text_dir = Path(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    relation_text_path = text_dir / "relation2text.txt"

    if not relation_text_path.exists():
        print("[relation-text] downloading relation descriptions...")
        urlretrieve(RELATION_TEXT_URL, relation_text_path)

    return relation_text_path


def clean_relation_text(text: str) -> str:
    """
    Normalize relation text.
    """
    text = text.strip()
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    return " ".join(text.split())


def read_relation_text_mapping(
    relation_text_path: str | Path,
) -> dict[str, str]:
    """
    Read:
        relation_key<TAB>relation_text
    """
    relation_text_path = Path(relation_text_path)

    mapping: dict[str, str] = {}

    with relation_text_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t", maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid relation-text line {line_number}: {line!r}"
                )

            relation_key, relation_text = parts

            relation_key = relation_key.strip()
            relation_text = clean_relation_text(relation_text)

            if not relation_key:
                raise ValueError(
                    f"Missing relation key on line {line_number}"
                )

            if not relation_text:
                raise ValueError(
                    f"Missing relation text for {relation_key!r}"
                )

            mapping[relation_key] = relation_text

    return mapping


def align_relation_texts(
    relation2id: dict[str, int],
    relation_text_mapping: dict[str, str],
) -> RelationTextAlignment:
    """
    Align relation text to the EXISTING relation2id mapping.

    No new relation IDs are created.
    texts_by_id[i] corresponds exactly to relation ID i.
    """
    texts_by_id = [""] * len(relation2id)

    matched = 0
    missing = 0

    for relation_key, relation_id in relation2id.items():
        relation_text = relation_text_mapping.get(relation_key)

        if relation_text:
            texts_by_id[relation_id] = relation_text
            matched += 1
        else:
            fallback = relation_key.replace("/", " ")
            fallback = fallback.replace("_", " ")
            fallback = fallback.replace(".", " ")
            fallback = " ".join(fallback.split())

            texts_by_id[relation_id] = fallback
            missing += 1

    return RelationTextAlignment(
        texts_by_id=texts_by_id,
        matched=matched,
        missing=missing,
    )