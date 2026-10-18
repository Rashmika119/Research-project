"""CPU contracts and derivatives only: no optimizer steps or training loops."""
import copy
import random
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch

from tests.helpers import fake_lm, make_tokens
from training.variants import build_scratch_model, configuration_settings, CONFIGURATIONS
from training.research_pipeline import trainable_state, load_model_state, completed_result
from training.experiment_io import save_checkpoint, file_digest, write_json
from training.reproducibility import (seed_everything, assert_exact_state, capture_rng,
                                      numerical_policy, restore_training_state)
from training.final_study import make_config, build_commands


class SelectedModelTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_layer_counts_scratch_initialization_and_shared_architecture(self, _):
        models = []
        for identifier in CONFIGURATIONS:
            architecture, _ = configuration_settings(identifier)
            seed_everything(0)
            with patch('torch.load', side_effect=AssertionError('Scratch may not load a checkpoint')):
                model = build_scratch_model(architecture, 4, 2)
            models.append(model)
        baseline, shared = models
        self.assertEqual(len(baseline.encoder.layers), 2)
        self.assertFalse(hasattr(baseline, 'bridge'))
        self.assertEqual(len(shared.structural.encoder.layers), 2)
        self.assertEqual(shared.structural.arrangement, 'c')
        self.assertTrue(shared.refinement_gated)
        self.assertFalse(shared.soft_prompt)
        self.assertFalse(hasattr(shared.bridge, 'kg_to_lm'))
        assert_exact_state(baseline.state_dict(), shared.structural.state_dict())
        for retired in ('original', 'residual-no-softprompt', 'residual', 'shared-a-gated', 'shared-b-gated', 'shared-c-ungated'):
            with self.assertRaises(ValueError):
                build_scratch_model(retired, 4, 2)
        from models.shared_rgat import SharedStructuralRGAT, SharedRGATRefinement
        with self.assertRaises(ValueError):
            SharedStructuralRGAT(4, 2, arrangement='a')
        with self.assertRaises(ValueError):
            SharedRGATRefinement(shared.structural, refinement_gated=False)
        from training.model_factory import build_model
        for override in ({'encoder_type': 'rgcn'}, {'scorer_type': 'distmult'}):
            with self.assertRaises(ValueError):
                build_model({**make_config('baseline', 0).model, **override}, 4, 2)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_cached_fusion_refinement_equations_and_projection_gradients(self, _):
        seed_everything(7)
        model = build_scratch_model('shared-c-gated', 4, 2).eval()
        edges = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]])
        types = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
        et, rt = make_tokens(4), make_tokens(2)
        ep, rp = model.bridge.lm.encode_text(**et), model.bridge.lm.encode_text(**rt)
        frozen = copy.deepcopy(model.bridge.lm.state_dict())
        from training.text_cache import pooled_text
        with tempfile.TemporaryDirectory() as directory:
            cached, key = pooled_text(model.bridge.lm, et, 2, 'cpu', directory)
            with patch.object(model.bridge.lm, 'encode_text', side_effect=AssertionError('Cached LM reencoded')):
                again, again_key = pooled_text(model.bridge.lm, et, 2, 'cpu', directory)
                assert_exact_state(cached, again)
                self.assertEqual(key, again_key)
                actual = model(edges, types, torch.arange(4), torch.arange(2), et, rt, edges,
                               text_batch_size=2, entity_pooled=ep, relation_pooled=rp)
        base = model.structural.encode(edges, types)
        u = base + model.entity_text_gate.tanh() * model.bridge(pooled=ep)
        expected_e = u + model.refinement_gate.tanh() * model.structural.refine(u, edges, types)
        expected_r = model.structural.scorer.relation_emb.weight + model.relation_text_gate.tanh() * model.bridge(pooled=rp)
        torch.testing.assert_close(actual[0], expected_e)
        torch.testing.assert_close(actual[1], expected_r)
        (actual[0].square().mean() + actual[1].square().mean()).backward()
        for module in (model.bridge.lm_to_kg, model.structural.encoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
        for gate in (model.entity_text_gate, model.relation_text_gate, model.refinement_gate):
            self.assertIsNotNone(gate.grad)
            self.assertTrue(torch.isfinite(gate.grad))
        self.assertTrue(all(p.grad is None for p in model.bridge.lm.parameters()))
        assert_exact_state(frozen, model.bridge.lm.state_dict())

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_checkpoint_state_optimizer_rng_and_identity_without_training(self, _):
        for architecture in ('baseline', 'shared-c-gated'):
            seed_everything(5)
            model = build_scratch_model(architecture, 4, 2)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
            # Fabricated nonzero moments exercise restoration without taking a step.
            for group in optimizer.param_groups:
                for p in group['params']:
                    optimizer.state[p] = {'step': torch.tensor(9.), 'exp_avg': torch.ones_like(p)*.02,
                                          'exp_avg_sq': torch.ones_like(p)*.003}
            negative, shuffle = random.Random(3), torch.Generator().manual_seed(4)
            saved = {'checkpoint_boundary': 'epoch_end', 'epoch': 1, 'optimizer_steps': 9,
                     'history': [{'epoch': 1, 'optimizer_steps': 9}],
                     'scheduler_state': None, 'grad_scaler_state': None,
                     'numerical_policy': numerical_policy(), 'model_state': trainable_state(model),
                     'optimizer_state': optimizer.state_dict(), 'rng': capture_rng(negative, shuffle),
                     'module_modes': {n: m.training for n, m in model.named_modules()}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'last.pt'
                save_checkpoint(path, saved)
                clone = build_scratch_model(architecture, 4, 2)
                adam = torch.optim.Adam((p for p in clone.parameters() if p.requires_grad), lr=.01)
                loaded = torch.load(path, weights_only=True)
                audit = restore_training_state(clone, adam, loaded, negative, shuffle)
                self.assertTrue(audit['optimizer_exact'] and audit['rng_exact'])
                assert_exact_state(trainable_state(model), trainable_state(clone))
                bad = copy.deepcopy(loaded)
                bad['optimizer_steps'] += 1
                with self.assertRaises(ValueError):
                    restore_training_state(clone, adam, bad, negative, shuffle)
                bad = copy.deepcopy(loaded['model_state']); bad.pop(next(iter(bad)))
                with self.assertRaises(ValueError):
                    load_model_state(clone, bad)
                result = {'run_id': 'matching', 'status': 'complete', 'checkpoint_sha256': {'last.pt': file_digest(path)}}
                write_json(Path(directory)/'results.json', result)
                self.assertEqual(completed_result(directory, 'matching'), result)
                with self.assertRaises(ValueError):
                    completed_result(directory, 'different-source-or-settings')
                path.write_bytes(b'corrupt fixture')
                with self.assertRaises(ValueError):
                    completed_result(directory, 'matching')
