"""Shared induced-subset graph/provenance checks; no training."""
from copy import deepcopy
from pathlib import Path
import unittest
from preprocessing.dataset import KGDataset
from preprocessing.induced_subset import make_induced_subset, graph_statistics
from training.experiment_io import prepare_data
from tests.test_research_pipeline import tiny_data

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

    def test_real_5000_when_downloaded(self):
        if not Path('data/raw/fb15k237/train.txt').exists():
            self.skipTest('Real dataset not downloaded; synthetic leakage checks still run')
        data, graph, known, manifest = prepare_data('data/raw/fb15k237', 5000, 0)
        self.assertEqual(data.num_entities, 5000)
        self.assertEqual(set(data.train), set(known))
        self.assertGreater(len(data.train), 5000)
        self.assertEqual(len(graph.edge_type), 2 * len(data.train))
        self.assertEqual(manifest['statistics']['entities'], 5000)


