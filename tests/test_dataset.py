"""Shared induced-subset graph/provenance checks; no training."""
from copy import deepcopy
from pathlib import Path
import unittest
from preprocessing.dataset import KGDataset
from preprocessing.induced_subset import make_induced_subset, graph_statistics
from training.experiment_io import prepare_data
from tests.helpers import tiny_data

class SubsetTests(unittest.TestCase):
    def test_induced_complete_exact_count_and_heldout_independence(self):
        full = tiny_data()
        data, graph, known, manifest = prepare_data('', 10, 0, full_dataset=full)
        old_ids = {full.entity2id[name] for name in data.entity2id}
        self.assertEqual(data.num_entities, 10)
        covered = [(h, r, t) for h, r, t in full.train if h in old_ids and t in old_ids]
        self.assertEqual(len(data.train), len(covered))
        self.assertEqual(set(data.train), set(known))
        self.assertEqual(len(graph.edge_type), 2 * len(covered))
        changed = deepcopy(full)
        changed.valid, changed.test = list(reversed(full.test)), list(reversed(full.valid))
        again = make_induced_subset(changed, 10, 0)
        self.assertEqual(data.entity2id, again.entity2id)
        self.assertEqual(data.train, again.train)
        self.assertEqual(data.relation2id, full.relation2id)
        with self.assertRaises(ValueError):
            make_induced_subset(full, full.num_entities + 1)
        self.assertEqual(manifest['selection_strategy'], 'train_bfs_induced_v1')

    def test_disconnected_training_components_fill_exactly(self):
        data = KGDataset({str(i): i for i in range(6)}, {'r': 0},
                         [(0, 0, 1), (2, 0, 3), (4, 0, 5)], [], [])
        subset = make_induced_subset(data, 5, 0)
        self.assertEqual(subset.num_entities, 5)
        self.assertEqual(graph_statistics(subset)['components'], 3)

    def test_full_original_dataset_when_available(self):
        if not all((Path('data/raw/fb15k237') / f'{split}.txt').exists()
                   for split in ('train', 'valid', 'test')):
            self.skipTest('Full dataset not present; no download in offline checks')
        from training.final_study import verify_manifest
        try:
            data, graph, known, manifest = prepare_data('data/raw/fb15k237', 0, 0)
        except PermissionError as error:
            self.skipTest(f'Local dataset unreadable in this environment: {error}')
        verify_manifest(manifest)
        self.assertEqual(len(graph.edge_type), 544230)
        self.assertEqual(set(data.train), set(known))
