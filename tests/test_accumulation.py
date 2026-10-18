"""Gradient equivalence and batching checks without optimizer steps or training."""
import copy
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import torch

from training.accumulation import backward_effective_batch
from training.baseline_fb15k import make_config
from training.losses import bce_loss
from training.negative_sampling import sample_negatives
from training.variants import build_scratch_model


class AccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_and_partial_batch_gradients_match_with_dropout(self):
        # Full effective window:8 micro-batches. Partial window:6, last is smaller.
        # Also exercise no accumulation and a one-example remainder.
        for count,micro in ((16,2),(11,2),(16,16),(1,2)):
            with self.subTest(count=count,micro=micro):
                torch.manual_seed(0)
                cfg=make_config()
                with patch('models.frozen_lm.load_pretrained_model',side_effect=AssertionError('No LM')):
                    reference=build_scratch_model('baseline',4,2,cfg.model).double().train()
                accumulated=copy.deepcopy(reference)
                before=copy.deepcopy(accumulated.state_dict())
                edges=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,0,0,1,2,3]])
                types=torch.tensor([0,1,0,1,2,3,2,3])
                positives=torch.tensor([[0,0,1],[1,1,2],[2,0,3],[3,1,0]]*4)[:count]
                negatives=sample_negatives(positives,4,set(map(tuple,positives.tolist())),10,random.Random(0))
                torch.manual_seed(123)
                e=reference.encode(edges,types)
                expected=bce_loss(reference.score_triples(e,positives),
                    reference.score_triples(e,negatives.reshape(-1,3)).reshape(count,10))
                expected.backward()
                expected_rng=torch.get_rng_state().clone()
                torch.manual_seed(123)
                calls=[]
                def encode():
                    calls.append(True)
                    return accumulated.encode(edges,types),accumulated.scorer.relation_emb.weight
                sizes=[]
                def observe(p,n):
                    sizes.append(len(p))
                    return bce_loss(p,n)
                with patch('training.accumulation.bce_loss',side_effect=observe):
                    actual=backward_effective_batch(encode,positives,negatives,micro,'cpu')
                self.assertEqual(len(calls),1)
                self.assertEqual(sum(sizes),count)
                self.assertLessEqual(max(sizes),micro)
                self.assertAlmostEqual(actual,expected.item(),places=12)
                self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
                for (name,a),(other,b) in zip(reference.named_parameters(),accumulated.named_parameters()):
                    self.assertEqual(name,other)
                    if a.grad is None:
                        self.assertIsNone(b.grad)
                    else:
                        torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-11)
                # Backward is permitted in this test; parameter updates are not.
                for name,value in accumulated.state_dict().items():
                    torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_batch_contract_and_final_window(self):
        cfg=make_config()
        self.assertEqual(cfg.train_batch_size,cfg.micro_batch_size*cfg.accumulation_steps)
        windows=[min(32768,483142-start) for start in range(0,483142,32768)]
        micro_counts=[len(range(0,n,4096)) for n in windows]
        self.assertEqual(micro_counts,[8]*14+[6])
        self.assertEqual(sum(micro_counts),118)
        self.assertEqual(windows[-1],5*4096+3910)
        for bad in (replace(cfg,accumulation_steps=7),replace(cfg,micro_batch_size=0),
                    replace(cfg,micro_batch_size=-1),replace(cfg,accumulation_steps=0)):
            with self.assertRaises(ValueError):bad.validate()

    def test_nonfinite_loss_fails_before_encoder_backward(self):
        entities=torch.randn(3,4,requires_grad=True)
        relations=torch.randn(1,4,requires_grad=True)
        triples=torch.tensor([[0,0,1],[1,0,2]])
        with patch('training.accumulation.bce_loss',return_value=torch.tensor(float('nan'))):
            with self.assertRaisesRegex(RuntimeError,'Non-finite'):
                backward_effective_batch(lambda:(entities,relations),triples,triples[:,None,:],1,'cpu')
        self.assertIsNone(entities.grad)
        self.assertIsNone(relations.grad)
