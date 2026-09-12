"""Derive a small connected subgraph from a full KGDataset for fast iteration.

This samples real entities/relations (unlike `dataset.load_synthetic_toy_graph`,
which fabricates a graph and is only used when no real dataset is available
at all). Use this once WN18 has loaded successfully, to smoke-test model
code against real data shape and relation semantics without waiting on a
full ~87k-triple training pass.
"""

from __future__ import annotations

import random
from collections import deque

from preprocessing.dataset import IdTriple, KGDataset


def make_toy_subset(
    dataset: KGDataset,
    max_entities: int = 50,
    seed: int = 0,
) -> KGDataset:
    """BFS-sample a connected chunk of `dataset.train` up to `max_entities`.

    Sampled entities/relations are remapped to a fresh contiguous 0..k-1 id
    range (a fresh `entity2id`), so the returned `KGDataset` is fully
    self-contained and `num_entities`/`num_relations` are always correct for
    the triples it actually contains — no stale ids from the parent dataset.
    Valid/test triples are kept only if both endpoints made it into the
    sampled entity set.
    """
    rng = random.Random(seed)

    adjacency: dict[int, list[IdTriple]] = {}
    for h, r, t in dataset.train:
        adjacency.setdefault(h, []).append((h, r, t))
        adjacency.setdefault(t, []).append((h, r, t))

    if not adjacency:
        raise ValueError("dataset.train is empty; cannot build a toy subset")

    start = rng.choice(list(adjacency.keys()))
    visited_entities: set[int] = {start}
    queue: deque[int] = deque([start])
    sampled_train: list[IdTriple] = []
    seen_triples: set[IdTriple] = set()

    while queue and len(visited_entities) < max_entities:
        node = queue.popleft()
        for triple in adjacency.get(node, []):
            h, r, t = triple
            other = t if node == h else h
            if len(visited_entities) >= max_entities and other not in visited_entities:
                continue  # would exceed the cap without also closing a loop
            if triple not in seen_triples:
                seen_triples.add(triple)
                sampled_train.append(triple)
            if other not in visited_entities:
                visited_entities.add(other)
                queue.append(other)

    def keep_if_covered(triples: list[IdTriple]) -> list[IdTriple]:
        return [
            (h, r, t)
            for h, r, t in triples
            if h in visited_entities and t in visited_entities
        ]

    id_to_entity_name = {v: k for k, v in dataset.entity2id.items()}
    remap = {orig: new for new, orig in enumerate(sorted(visited_entities))}
    subset_entity2id = {
        id_to_entity_name[orig]: new_id for orig, new_id in remap.items()
    }

    def remap_triples(triples: list[IdTriple]) -> list[IdTriple]:
        return [(remap[h], r, remap[t]) for h, r, t in triples]

    return KGDataset(
        entity2id=subset_entity2id,
        relation2id=dict(dataset.relation2id),
        train=remap_triples(sampled_train),
        valid=remap_triples(keep_if_covered(dataset.valid)),
        test=remap_triples(keep_if_covered(dataset.test)),
    )
