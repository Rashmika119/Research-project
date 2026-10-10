"""Use real RGAT/ComplEx with a tiny deterministic stand-in for the frozen LM.

These tests verify mechanics, not RoBERTa quality or research performance.
The real Hugging Face path has a separate opt-in smoke runner.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluation.enriched_scorer import EnrichedScorer
from models.frozen_lm import FrozenLM
from models.scorer import ComplExScorer
from preprocessing.dataset import load_synthetic_toy_graph
from preprocessing.graph_builder import build_train_graph
from training.experiment_io import prepare_data, write_json
from training.reports import summarize, comparison_report, final_comparison
from training.research_pipeline import (ExperimentConfig, load_model_state, run_experiment,
                                        state_fingerprint, trainable_state)
from training.text_cache import pooled_text
from training.variants import MODEL_CONFIG, VARIANTS, build_scratch_model, warmup_epochs
from training.reproducibility import assert_exact_state


class TinyFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, _commit_hash='offline-test-revision')
        with torch.random.fork_rng():
            torch.manual_seed(987)
            self.embedding = nn.Embedding(64, 8)
            self.linear = nn.Linear(8, 8)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        self.calls.append({'input_ids': input_ids is not None,
                           'inputs_embeds': inputs_embeds is not None,
                           'length': attention_mask.shape[1]})
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        context = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.tanh(self.linear(x) + context))


def fake_lm(*args, **kwargs):
    return TinyFrozenLM()


def make_tokens(count):
    ids = (torch.arange(count * 5).reshape(count, 5) % 60) + 1
    mask = torch.ones_like(ids)
    mask[:, -2:] = 0
    return {'input_ids': ids, 'attention_mask': mask}


def tiny_data():
    return load_synthetic_toy_graph(num_entities=12, num_relations=3,
                                    num_train=35, num_valid=5, num_test=5, seed=3)


def small_config(**kwargs):
    return ExperimentConfig(max_entities=0, epochs=3, warmup=1,
                            model={**MODEL_CONFIG, 'dim': 8}, lm_name='offline-test', **kwargs)


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_all_five_forward_backward_and_frozen_weights(self, _):
        data = tiny_data()
        graph = build_train_graph(data.train, data.num_entities, data.num_relations)
        edges = torch.tensor(graph.edge_index).t().contiguous()
        types = torch.tensor(graph.edge_type)
        et, rt = make_tokens(data.num_entities), make_tokens(data.num_relations)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                # Any accidental checkpoint initialization fails immediately.
                with patch('torch.load', side_effect=AssertionError('No checkpoint allowed')):
                    model = build_scratch_model(variant, data.num_entities, data.num_relations,
                                                {**MODEL_CONFIG, 'dim': 8}, 'offline-test')
                before = state_fingerprint(model.bridge.lm)
                model.train()
                self.assertFalse(model.bridge.lm.lm.training)
                e, r = model(edges, types, torch.arange(data.num_entities),
                             torch.arange(data.num_relations), et, rt, edges, text_batch_size=4)
                self.assertEqual(e.shape, (data.num_entities, 8))
                self.assertEqual(r.shape, (data.num_relations, 8))
                triples = torch.tensor(data.train[:8])
                score = model.score_triples(e, r, triples)
                scorer = EnrichedScorer(r)
                tails = scorer.score_all_tails(e, e[triples[:, 0]], triples[:, 1])
                heads = scorer.score_all_heads(e, triples[:, 1], e[triples[:, 2]])
                torch.testing.assert_close(score, tails[torch.arange(8), triples[:, 2]])
                torch.testing.assert_close(score, heads[torch.arange(8), triples[:, 0]])
                loss = torch.nn.functional.softplus(-score).mean() + .01 * (e.square().mean() + r.square().mean())
                loss.backward()
                modules = [model.structural.encoder, model.structural.entity_emb,
                           model.structural.scorer.relation_emb, model.bridge.lm_to_kg]
                if model.soft_prompt:
                    modules.append(model.bridge.kg_to_lm)
                else:
                    self.assertFalse(hasattr(model.bridge, 'kg_to_lm'))
                    self.assertIsNotNone(model.refiner)
                    self.assertTrue(model.residual)
                    self.assertTrue(all(c == {'input_ids': True, 'inputs_embeds': False, 'length': 5}
                                        for c in model.bridge.lm.lm.calls))
                if model.refiner is not None:
                    modules.append(model.refiner)
                for module in modules:
                    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                    self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                    self.assertTrue(any(g.abs().sum() > 0 for g in grads))
                for name, param in model.named_parameters():
                    if name.endswith('_gate'):
                        self.assertIsNotNone(param.grad)
                        self.assertTrue(torch.isfinite(param.grad).all())
                        self.assertAlmostEqual(param.item(), .01, places=6)
                torch.optim.Adam((p for p in model.parameters() if p.requires_grad)).step()
                self.assertEqual(before, state_fingerprint(model.bridge.lm))
                self.assertTrue(all(p.grad is None for p in model.bridge.lm.parameters()))
                clone = build_scratch_model(variant, data.num_entities, data.num_relations,
                                             {**MODEL_CONFIG, 'dim': 8}, 'offline-test')
                load_model_state(clone, trainable_state(model))
                model.eval()
                clone.eval()
                with torch.no_grad():
                    args = (edges, types, torch.arange(data.num_entities), torch.arange(data.num_relations), et, rt, edges)
                    for a, b in zip(model(*args), clone(*args)):
                        torch.testing.assert_close(a, b)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_original_pair_identical_and_budget(self, _):
        states = []
        for variant in ('original', 'original-no-warmup'):
            torch.manual_seed(9)
            model = build_scratch_model(variant, 12, 3, {**MODEL_CONFIG, 'dim': 8}, 'offline-test')
            states.append(trainable_state(model))
            self.assertFalse(model.residual)
            self.assertTrue(model.soft_prompt)
            self.assertIsNotNone(model.refiner)
        self.assertEqual(states[0].keys(), states[1].keys())
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        self.assertEqual(warmup_epochs('original', 30, 5), 5)
        self.assertEqual(warmup_epochs('original-no-warmup', 30, 5), 0)
        with self.assertRaises(ValueError):
            warmup_epochs('original', 5, 5)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_pooling_cache_and_trainable_projection(self, _):
        model = build_scratch_model('residual-no-softprompt', 12, 3,
                                    {**MODEL_CONFIG, 'dim': 8}, 'offline-test')
        tokens = make_tokens(4)
        lm = model.bridge.lm
        hidden = lm.lm(**tokens).last_hidden_state
        expected = hidden[:, :3].mean(1)
        torch.testing.assert_close(lm.encode_text(**tokens), expected)
        changed = {k: v.clone() for k, v in tokens.items()}
        changed['input_ids'][:, -2:] = 0
        torch.testing.assert_close(lm.encode_text(**changed), expected)
        with self.assertRaises(ValueError):
            lm.encode_text(tokens['input_ids'], torch.zeros_like(tokens['attention_mask']))
        with tempfile.TemporaryDirectory() as directory:
            pooled, key = pooled_text(lm, tokens, 2, 'cpu', directory)
            calls = len(lm.lm.calls)
            cached, key2 = pooled_text(lm, tokens, 2, 'cpu', directory)
            self.assertEqual(key, key2)
            self.assertEqual(calls, len(lm.lm.calls))
            torch.testing.assert_close(pooled, cached)
            self.assertFalse(cached.requires_grad)
            model.eval()
            a = model.bridge(**tokens)
            b = model.bridge(pooled=cached)
            torch.testing.assert_close(a, b)
            initial = state_fingerprint(model.bridge.lm_to_kg)
            b.square().mean().backward()
            torch.optim.Adam(model.bridge.lm_to_kg.parameters(), lr=.01).step()
            self.assertNotEqual(initial, state_fingerprint(model.bridge.lm_to_kg))
            altered = {k: v.clone() for k, v in tokens.items()}
            altered['input_ids'][0, 0] += 1
            _, new_key = pooled_text(lm, altered, 2, 'cpu', directory)
            self.assertNotEqual(key, new_key)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manifest_train_edges_and_split_provenance(self):
        full = tiny_data()
        data, graph, known, manifest = prepare_data('', 10, 0, full_dataset=full)
        inv_names = {v: k for k, v in data.entity2id.items()}
        def original(triples):
            return {(full.entity2id[inv_names[h]], r, full.entity2id[inv_names[t]]) for h, r, t in triples}
        self.assertTrue(original(data.train) <= set(full.train))
        self.assertTrue(original(data.valid) <= set(full.valid))
        self.assertTrue(original(data.test) <= set(full.test))
        direct = {(h, r, t) for (h, t), r in zip(graph.edge_index, graph.edge_type) if r < data.num_relations}
        self.assertEqual(direct, set(data.train))
        self.assertTrue(set(data.train) <= set(known))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_json(path, manifest)
            prepare_data('', 10, 0, path, full_dataset=full)
            changed = tiny_data()
            changed.train = list(reversed(changed.train))
            with self.assertRaises(ValueError):
                prepare_data('', 10, 0, path, full_dataset=changed)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_train_baseline_and_all_variants_skip_and_outputs(self, _):
        data = tiny_data()
        tokens = (make_tokens(data.num_entities), make_tokens(data.num_relations))
        fingerprints = []
        with tempfile.TemporaryDirectory() as directory:
            for variant in ('baseline', *VARIANTS):
                cfg = small_config(variant=variant)
                folder = Path(directory) / variant
                result = run_experiment(cfg, folder, full_dataset=data, tokens=tokens)
                self.assertEqual(result['optimizer_steps'], 3)
                self.assertFalse(result['baseline_checkpoint_used'])
                self.assertEqual(result['ranking_policy'], 'average_exact_ties')
                self.assertTrue(all((folder / f).exists() for f in (
                    'best.pt', 'last.pt', 'config.json', 'history.json', 'results.json', 'history.csv', 'metrics.csv')))
                self.assertEqual(result['warmup_epochs'], 0 if variant in ('baseline', 'original-no-warmup') else 1)
                self.assertEqual(result['main_epochs'] + result['warmup_epochs'], 3)
                fingerprints.append(result['initial_structural_sha256'])
                saved = torch.load(folder / 'best.pt', weights_only=True)
                self.assertFalse(any(k.startswith('bridge.lm.lm.') for k in saved['model_state']))
                again = run_experiment(cfg, folder, full_dataset=data, tokens=tokens, skip_completed=True)
                self.assertEqual(again['run_id'], result['run_id'])
                with self.assertRaises(ValueError):
                    run_experiment(replace(cfg, lr=.02), folder, full_dataset=data, tokens=tokens, skip_completed=True)
            self.assertEqual(len(set(fingerprints)), 1)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_resume_matches_uninterrupted(self, _):
        data = tiny_data()
        tokens = (make_tokens(data.num_entities), make_tokens(data.num_relations))
        with tempfile.TemporaryDirectory() as directory:
            cfg = small_config(variant='original')
            clean = Path(directory) / 'clean'
            interrupted = Path(directory) / 'interrupted'
            expected = run_experiment(cfg, clean, full_dataset=data, tokens=tokens)
            from training import research_pipeline
            real_save = research_pipeline.save_checkpoint
            def save_then_interrupt(path, value):
                real_save(path, value)
                if Path(path).name == 'last.pt' and value['epoch'] == 2:
                    raise RuntimeError('simulated runtime disconnect')
            with patch('training.research_pipeline.save_checkpoint', side_effect=save_then_interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    run_experiment(cfg, interrupted, full_dataset=data, tokens=tokens)
            actual = run_experiment(cfg, interrupted, full_dataset=data, tokens=tokens, resume=True)
            self.assertEqual(expected['validation'], actual['validation'])
            self.assertEqual(expected['test'], actual['test'])
            self.assertEqual(expected['best_epoch'], actual['best_epoch'])
            a = torch.load(clean / 'last.pt', weights_only=True, map_location='cpu')
            b = torch.load(interrupted / 'last.pt', weights_only=True, map_location='cpu')
            for key in ('model_state', 'optimizer_state', 'rng', 'epoch', 'optimizer_steps',
                        'best_model_state', 'best_epoch', 'initial_validation', 'module_modes'):
                assert_exact_state(a[key], b[key], key)
            for expected_row, actual_row in zip(expected['history'], actual['history']):
                assert_exact_state({k: v for k, v in expected_row.items() if k != 'epoch_seconds'},
                                   {k: v for k, v in actual_row.items() if k != 'epoch_seconds'})
            audit = json.loads((interrupted / 'resume_audit.json').read_text())
            self.assertTrue(audit['optimizer_exact'] and audit['rng_exact'] and audit['model_and_buffers_exact'])

    def test_selection_uses_validation_and_requires_all_runs(self):
        results = []
        for index, variant in enumerate(VARIANTS):
            for seed in (0, 1, 2):
                cfg = small_config(variant=variant, seed=seed)
                from dataclasses import asdict
                result = {'status': 'complete', 'initialization': 'scratch', 'settings': asdict(cfg),
                    'manifest_sha256': 'same', 'ranking_policy': 'average_exact_ties',
                    'initial_structural_sha256': f'seed{seed}', 'best_epoch': 2,
                    'optimizer_steps': 3, 'warmup_epochs': 0 if variant == 'original-no-warmup' else 1,
                    'main_epochs': 3 if variant == 'original-no-warmup' else 2,
                    'duration_seconds': 1., 'peak_gpu_memory_bytes': None,
                    'validation': {m: .1 + index * .05 + seed * .01 for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')},
                    'test': {m: .9 - index * .05 for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')},
                    'history': [{'epoch': 2, 'phase': 'main', 'loss': 1.}]}
                results.append(result)
        summary, _ = summarize(results)
        self.assertEqual(summary['selected_by_validation'], 'residual-no-softprompt')
        self.assertAlmostEqual(summary['variants']['original']['validation']['MRR']['std'], .01)
        with self.assertRaises(ValueError):
            summarize(results[:-1])
        with tempfile.TemporaryDirectory() as directory:
            comparison_report(directory, results)
            for name in ('summary.json', 'runs.json', 'runs.csv', 'comparison.csv',
                         'training_curves.png', 'validation_comparison.png'):
                self.assertTrue((Path(directory) / name).is_file())
            with self.assertRaises(ValueError):
                final_comparison({'dataset_scope': 'subset'}, {'dataset_scope': 'full'}, directory)


if __name__ == '__main__':
    unittest.main()
