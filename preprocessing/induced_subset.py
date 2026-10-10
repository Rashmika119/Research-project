"""Deterministic train-only BFS entity selection and ALL induced training facts."""
from collections import deque
import random

from preprocessing.dataset import KGDataset

STRATEGY = 'train_bfs_induced_v1'


def make_induced_subset(dataset, max_entities, seed=0):
    if not 1 <= max_entities <= dataset.num_entities:
        raise ValueError('Requested entity count must fit the dataset; never silently reduce it')
    adjacency = {}
    for h, r, t in dataset.train:
        adjacency.setdefault(h, []).append(t)
        adjacency.setdefault(t, []).append(h)
    if not adjacency:
        raise ValueError('No training facts for entity selection')
    rng = random.Random(seed)
    # Same first root and adjacency order as the historical BFS sampler.
    start = rng.choice(list(adjacency))
    selected, queue = {start}, deque([start])
    training_entities = sorted(adjacency)
    while len(selected) < max_entities:
        if not queue:
            remaining = [entity for entity in training_entities if entity not in selected]
            if not remaining:
                raise ValueError('Not enough entities incident to training facts; refusing held-out-based selection')
            root = rng.choice(remaining)
            selected.add(root)
            queue.append(root)
        node = queue.popleft()
        for other in adjacency[node]:
            if other not in selected and len(selected) < max_entities:
                selected.add(other)
                queue.append(other)
    remap = {old: new for new, old in enumerate(sorted(selected))}
    names = {v: k for k, v in dataset.entity2id.items()}
    def induced(triples):
        return [(remap[h], r, remap[t]) for h, r, t in triples if h in selected and t in selected]
    return KGDataset({names[old]: new for old, new in remap.items()}, dict(dataset.relation2id),
                     induced(dataset.train), induced(dataset.valid), induced(dataset.test))


def graph_statistics(data):
    neighbors = [set() for _ in range(data.num_entities)]
    for h, _, t in data.train:
        if h != t:
            neighbors[h].add(t)
            neighbors[t].add(h)
    unseen = set(range(data.num_entities))
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue, size = [root], 0
        while queue:
            node = queue.pop()
            size += 1
            for other in neighbors[node]:
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        components.append(size)
    pairs = sum(map(len, neighbors)) // 2
    possible = data.num_entities * (data.num_entities - 1) / 2
    return {'entities': data.num_entities, 'relations': data.num_relations,
            'train_triples': len(data.train), 'valid_triples': len(data.valid), 'test_triples': len(data.test),
            'isolated_entity_ids': [i for i, adjacent in enumerate(neighbors) if not adjacent],
            'components': len(components), 'component_sizes': sorted(components, reverse=True),
            'unique_undirected_pairs': pairs, 'undirected_density': pairs / possible if possible else 0.,
            'mean_undirected_degree': 2 * pairs / data.num_entities,
            'train_triples_per_entity': len(data.train) / data.num_entities,
            'definition': 'Weak connectivity; undirected simple density excludes self-loops and duplicate relation pairs'}
