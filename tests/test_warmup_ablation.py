"""Warmup factorial design, induced data, exact resume and descriptive reporting."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from preprocessing.dataset import KGDataset
from preprocessing.induced_subset import make_induced_subset, graph_statistics
from training.ablation_reports import ablation_report, validate_runs, PLOTS, analyze_run
from training.experiment_io import prepare_data
from training.research_pipeline import run_experiment
from training.reproducibility import assert_exact_state
from training.variants import CONFIGURATIONS, configuration_settings
from tests.test_notebook import fixture_result
from tests.test_research_pipeline import tiny_data, make_tokens, small_config, fake_lm


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


class WarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_four_full_budgets_scratch_initialization_and_identity_rejection(self, _):
        data = tiny_data()
        tokens = make_tokens(data.num_entities), make_tokens(data.num_relations)
        results = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for identifier in CONFIGURATIONS:
                architecture, warm = configuration_settings(identifier)
                cfg = replace(small_config(variant=architecture), warmup=warm, epochs=30, max_entities=12)
                output = root / identifier
                r = run_experiment(cfg, output, manifest_path=root / 'manifest.json',
                                   cache_dir=root / 'text_cache', full_dataset=data, tokens=tokens)
                results.append(r)
                self.assertEqual(r['configuration'], identifier)
                self.assertEqual(r['optimizer_steps'], 30)
                self.assertEqual(r['structural_optimizer_steps'], warm)
                self.assertEqual(r['integrated_optimizer_steps'], 30 - warm)
                self.assertEqual(sum(h['phase'] == 'structural_warmup' for h in r['history']), warm)
                self.assertTrue(all('Hits@10' in h['validation'] for h in r['history']))
                for changed in (replace(cfg, warmup=0 if warm else 5), replace(cfg, seed=2),
                                replace(cfg, variant='no-refinement'), replace(cfg, subset_seed=1)):
                    with self.assertRaises(ValueError):
                        run_experiment(changed, output, full_dataset=data, tokens=tokens, resume=True)
            self.assertEqual(len({r['manifest_sha256'] for r in results}), 1)
            self.assertEqual(len({r['initial_structural_sha256'] for r in results}), 1)
            for architecture in ('residual', 'residual-no-softprompt'):
                self.assertEqual(len({r['initial_trainable_sha256'] for r in results if r['architecture'] == architecture}), 1)
            self.assertEqual(results[2]['text_cache_keys'], results[3]['text_cache_keys'])

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_all_four_resume_paths_and_swapped_checkpoint_rejection(self, _):
        from training import research_pipeline as pipeline
        data = tiny_data()
        tokens = make_tokens(data.num_entities), make_tokens(data.num_relations)
        with tempfile.TemporaryDirectory() as directory:
            for identifier in CONFIGURATIONS:
                architecture, warm = configuration_settings(identifier, 1)
                cfg = replace(small_config(variant=architecture), warmup=warm)
                clean, resumed = [Path(directory) / identifier / name for name in ('clean', 'resumed')]
                run_experiment(cfg, clean, full_dataset=data, tokens=tokens)
                actual_save = pipeline.save_checkpoint
                def interrupt(path, state):
                    actual_save(path, state)
                    if Path(path).name == 'last.pt' and state['epoch'] == 2:
                        raise InterruptedError('fixture disconnect')
                with patch.object(pipeline, 'save_checkpoint', side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        run_experiment(cfg, resumed, full_dataset=data, tokens=tokens)
                good = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=True)
                bad = deepcopy(good)
                bad['run_id'] = 'another_configuration'
                actual_save(resumed / 'last.pt', bad)
                with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
                    run_experiment(cfg, resumed, full_dataset=data, tokens=tokens, resume=True)
                actual_save(resumed / 'last.pt', good)
                run_experiment(cfg, resumed, full_dataset=data, tokens=tokens, resume=True)
                left, right = [torch.load(p / 'last.pt', weights_only=True, map_location='cpu') for p in (clean, resumed)]
                for key in ('model_state', 'optimizer_state', 'rng', 'epoch', 'optimizer_steps', 'best_model_state'):
                    assert_exact_state(left[key], right[key], key)


class ReportTests(unittest.TestCase):
    def test_all_reports_validation_selection_pairing_and_guards(self):
        results = [fixture_result(c, s) for c in CONFIGURATIONS for s in (0, 1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            summary = ablation_report(directory, results)
            self.assertEqual(summary['selected_by_validation'], 'residual-no-softprompt-no-warmup')
            self.assertEqual(len(summary['paired_comparisons']), 20)
            for name in (*PLOTS, 'summary.json', 'comparison.csv', 'runs.csv', 'training_behavior.json',
                         'paired_seed_differences.csv', 'paired_comparisons.json', 'ties.csv'):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            self.assertAlmostEqual(summary['configurations']['residual-warmup']['validation']['MRR']['std'], .01)
            self.assertAlmostEqual(summary['paired_comparisons'][0]['mean'], -.03)
        for field, value in (('manifest_sha256', 'other'), ('initial_trainable_sha256', 'changed'),
                             ('optimizer_steps', 31), ('candidate_entities', 11)):
            changed = deepcopy(results)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runs(changed, [0, 1, 2])
        with self.assertRaises(ValueError):
            validate_runs(results[:-1], [0, 1, 2])

    def test_spikes_plateau_and_tie_flags_are_descriptive(self):
        r = fixture_result('residual-no-warmup', 0)
        r['history'][0]['loss'], r['history'][1]['loss'], r['history'][2]['loss'] = 1.4, 1.6, 1.3
        r['validation']['Tie_query_fraction'] = .9
        behavior = analyze_run(r)
        self.assertTrue(behavior['severe_tie_flags'])
        self.assertTrue(behavior['loss_falls_while_ranking_stalls_last_up_to_5_main_epochs'])
        self.assertIsNone(behavior['warmup_loss'])
        self.assertEqual(behavior['integrated_loss']['upward_steps'], 1)
