"""FB15k-237 and original FB15K TSV loading with deterministic ID mapping.

Research callers select the dataset explicitly and validate its exact counts.
FB15K uses a separate directory and staged download, with no synthetic fallback.
Both parse head/relation/tail columns and assign IDs in train-first order.
Synthetic toy graphs are only for offline mechanics tests, never research results.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

Triple = tuple[str, str, str]
IdTriple = tuple[int, int, int]

# FB15k-237 mirrors in the same tab-separated train/valid/test.txt layout
# used by most KGE codebases (the same layout WN18, WN18RR, etc. also use).
# Tried in order, first one that serves all three files wins. NONE of these
# have been verified from this dev machine (no network access here) — one of
# these same repos' WN18 folder already 404'd once and needed a fallback
# mirror in practice, so don't be surprised if this needs the same treatment.
# If every candidate below fails when you run the Phase 0 smoke test in
# Colab, either add another working mirror here or place
# train.txt/valid.txt/test.txt in `raw_dir` manually; `load_dataset(...,
# use_synthetic_fallback=True)` will still let you smoke-test the rest of
# the pipeline in the meantime.
_FB15K237_URL_CANDIDATES = [
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/",
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/main/FB15k-237/",
    "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/FB15k-237/",
]
_SPLIT_FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
FB15K_COUNTS = (14951, 1345, 483142, 50000, 59071)
FB15K_URL = 'https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k/'


@dataclass
class KGDataset:
    """A knowledge graph split into train/valid/test with shared ID maps."""

    entity2id: dict[str, int]
    relation2id: dict[str, int]
    train: list[IdTriple]
    valid: list[IdTriple]
    test: list[IdTriple]

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


def _download_split_files(base_url: str, dest_dir: Path) -> None:
    """Download all three split files from one mirror, all-or-nothing.

    Downloads into a temporary directory first and only moves the files
    into `dest_dir` once every one of them has succeeded — so a mirror that
    serves e.g. train.txt but 404s on valid.txt can't leave `dest_dir` with
    files mixed from two different mirrors.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for filename in _SPLIT_FILES.values():
            urllib.request.urlretrieve(base_url + filename, tmp_dir / filename)
        for filename in _SPLIT_FILES.values():
            shutil.move(str(tmp_dir / filename), str(dest_dir / filename))


def download_fb15k237(raw_dir: str | Path) -> Path:
    """Download FB15k-237 train/valid/test files into `raw_dir` if missing.

    Tries each mirror in `_FB15K237_URL_CANDIDATES` in order; the first one
    that serves all three files wins. Returns the directory containing them.
    Raises `RuntimeError` (listing every mirror that failed and why) if none
    of them work — callers should catch this and fall back to
    `load_synthetic_toy_graph` when offline.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all((raw_dir / f).exists() for f in _SPLIT_FILES.values()):
        return raw_dir  # already downloaded, nothing to do

    errors: list[str] = []
    for base_url in _FB15K237_URL_CANDIDATES:
        try:
            _download_split_files(base_url, raw_dir)
            break
        except Exception as exc:  # network unavailable, 404, blocked, etc.
            errors.append(f"{base_url} -> {exc}")
    else:
        raise RuntimeError(
            "Could not download FB15k-237 from any known mirror:\n  "
            + "\n  ".join(errors)
            + "\nPlace train.txt/valid.txt/test.txt in raw_dir manually, or "
            "call load_dataset(..., use_synthetic_fallback=True) for an "
            "offline smoke test."
        )

    missing = [f for f in _SPLIT_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing FB15k-237 files after download attempt: {missing}"
        )
    return raw_dir


def _read_triples(path: Path) -> list[Triple]:
    """Read one FB15k-237 split file and return (head, relation, tail)
    triples.

    Assumes the on-disk column order is already "head \\t relation \\t tail"
    (the conventional order) — this is UNVERIFIED for whichever mirror
    actually serves the files (see the module docstring). If
    `KGDataset.num_relations` comes out wildly wrong (should be 237) after
    a real download, inspect a raw line from the downloaded file and swap
    the unpacking order below to match, the same way this project had to
    fix WN18's non-obvious (head, tail, relation) column order.
    """
    triples: list[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_no}: expected 3 tab-separated fields, got "
                    f"{len(parts)}"
                )
            h, r, t = parts  # assumed on-disk order: (head, relation, tail)
            triples.append((h, r, t))
    return triples


def _build_id_maps(
    split_triples: dict[str, list[Triple]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Assign IDs in first-seen order, scanning train before valid/test.

    This anchors the vocabulary to the training graph. Any entities/relations
    that only appear in valid/test (rare, but not guaranteed) are appended
    afterwards, so every split can still be converted to valid IDs.
    """
    entity2id: dict[str, int] = {}
    relation2id: dict[str, int] = {}

    for split_name in ("train", "valid", "test"):
        for h, r, t in split_triples.get(split_name, []):
            for ent in (h, t):
                if ent not in entity2id:
                    entity2id[ent] = len(entity2id)
            if r not in relation2id:
                relation2id[r] = len(relation2id)

    return entity2id, relation2id


def load_fb15k237(
    raw_dir: str | Path, download_if_missing: bool = True
) -> KGDataset:
    """Load FB15k-237 from `raw_dir`, downloading it first if needed."""
    raw_dir = Path(raw_dir)
    if download_if_missing:
        download_fb15k237(raw_dir)

    split_str_triples = {
        split: _read_triples(raw_dir / filename)
        for split, filename in _SPLIT_FILES.items()
    }
    entity2id, relation2id = _build_id_maps(split_str_triples)

    def to_ids(triples: list[Triple]) -> list[IdTriple]:
        return [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

    return KGDataset(
        entity2id=entity2id,
        relation2id=relation2id,
        train=to_ids(split_str_triples["train"]),
        valid=to_ids(split_str_triples["valid"]),
        test=to_ids(split_str_triples["test"]),
    )


def validate_fb15k(data: KGDataset) -> None:
    actual = (data.num_entities, data.num_relations, len(data.train), len(data.valid), len(data.test))
    if actual != FB15K_COUNTS:
        raise ValueError(f'Expected original FB15K {FB15K_COUNTS}, received {actual}; no fallback')


def load_fb15k(raw_dir: str | Path, download_if_missing: bool = True) -> KGDataset:
    """Load original FB15K head/relation/tail TSV files; never use FB15k-237.

    Reuse the deterministic train-first ID mapping. Download all splits to a
    temporary directory, validate them, then publish only into an empty dataset
    directory. Never overwrite a partial or existing user dataset automatically.
    """
    raw_dir = Path(raw_dir)
    present = [(raw_dir/f).is_file() for f in _SPLIT_FILES.values()]
    if all(present):
        data = load_fb15k237(raw_dir, download_if_missing=False)  # Shared TSV parser only.
        validate_fb15k(data)
        return data
    if any(present):
        raise ValueError(f'Incomplete FB15K directory: {raw_dir}; provide all three splits or use an empty directory')
    if not download_if_missing:
        raise FileNotFoundError(f'FB15K splits missing: {raw_dir}')
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fb15k_') as temporary:
        staging = Path(temporary)
        try:
            _download_split_files(FB15K_URL, staging)
        except Exception as error:
            raise RuntimeError(f'FB15K download failed from {FB15K_URL}; provide train.txt/valid.txt/test.txt in {raw_dir}') from error
        data = load_fb15k237(staging, download_if_missing=False)
        validate_fb15k(data)
        for filename in _SPLIT_FILES.values():
            if (raw_dir/filename).exists():
                raise FileExistsError(f'Dataset appeared during download: {raw_dir/filename}')
        for filename in _SPLIT_FILES.values():
            shutil.move(str(staging/filename), str(raw_dir/filename))
    return data


def load_synthetic_toy_graph(
    num_entities: int = 40,
    num_relations: int = 5,
    num_train: int = 200,
    num_valid: int = 20,
    num_test: int = 20,
    seed: int = 0,
) -> KGDataset:
    """Deterministic fake KG with the same contract as `load_fb15k237`.

    Used only when the real dataset can't be obtained, to still smoke-test
    the ID-mapping / adjacency / leakage pipeline end to end. Never use this
    for reported results.
    """
    rng = random.Random(seed)
    entity2id = {f"e{i}": i for i in range(num_entities)}
    relation2id = {f"r{i}": i for i in range(num_relations)}

    # `seen` is shared across all three calls below so no (h, r, t) triple
    # can land in more than one split by random chance — otherwise
    # assert_no_leakage() could fail on a harmless synthetic-data coincidence
    # instead of a real bug.
    seen: set[IdTriple] = set()

    def sample_triples(n: int) -> list[IdTriple]:
        out: list[IdTriple] = []
        max_attempts = n * 50 + 1000
        for _ in range(max_attempts):
            if len(out) == n:
                break
            h = rng.randrange(num_entities)
            t = rng.randrange(num_entities)
            if t == h:
                continue
            r = rng.randrange(num_relations)
            triple = (h, r, t)
            if triple in seen:
                continue
            seen.add(triple)
            out.append(triple)
        if len(out) < n:
            raise RuntimeError(
                f"Could not sample {n} unique synthetic triples out of "
                f"{num_entities * (num_entities - 1) * num_relations} possible "
                "— increase num_entities/num_relations."
            )
        return out

    return KGDataset(
        entity2id=entity2id,
        relation2id=relation2id,
        train=sample_triples(num_train),
        valid=sample_triples(num_valid),
        test=sample_triples(num_test),
    )


def load_dataset(
    raw_dir: str | Path,
    download_if_missing: bool = True,
    use_synthetic_fallback: bool = True,
) -> KGDataset:
    """Try to load real FB15k-237; optionally fall back to a synthetic toy
    graph.

    Set `use_synthetic_fallback=False` once you've confirmed the real
    dataset loads (e.g. in CI or a from-scratch Colab run) so a silent
    network failure can't be mistaken for a successful real-data run.
    """
    try:
        return load_fb15k237(raw_dir, download_if_missing=download_if_missing)
    except RuntimeError as exc:
        if not use_synthetic_fallback:
            raise
        print(f"[dataset] Falling back to synthetic toy graph: {exc}")
        return load_synthetic_toy_graph()
