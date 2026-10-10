import copy
import random
import unittest

import numpy as np
import torch

from training.reproducibility import capture_rng, restore_rng, seed_everything, assert_exact_state


class ReproducibilityTests(unittest.TestCase):
    def test_all_rng_streams_round_trip(self):
        seed_everything(12)
        negative = random.Random(15)
        shuffle = torch.Generator().manual_seed(16)
        saved = capture_rng(negative, shuffle)
        def draw():
            return {'python': random.random(), 'numpy': np.random.random(),
                    'torch': torch.rand(5), 'negative': negative.random(),
                    'shuffle': torch.randperm(9, generator=shuffle),
                    'cuda': [torch.rand(5, device=f'cuda:{i}').cpu() for i in range(torch.cuda.device_count())]}
        expected = draw()
        draw()
        restore_rng(saved, negative, shuffle)
        assert_exact_state(saved, capture_rng(negative, shuffle))
        assert_exact_state(expected, draw())

    def test_divergent_weights_and_discrete_state_fail_exact_audit(self):
        expected = {'weight': torch.ones(3), 'step': torch.tensor(2), 'epoch': 2}
        for key in expected:
            actual = copy.deepcopy(expected)
            actual[key] += .001 if key == 'weight' else 1
            with self.subTest(key=key), self.assertRaises(AssertionError):
                assert_exact_state(expected, actual)
