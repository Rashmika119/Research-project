"""Explicit numerical policy and complete epoch-boundary RNG restoration."""
import os
import random

import numpy as np
import torch


def numerical_policy():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
            'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp': torch.backends.cuda.math_sdp_enabled()}


def seed_everything(seed, deterministic=True):
    # Must be set before the first cuBLAS operation; CLI/Colab also set it at startup.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def capture_rng(negative_rng, shuffle_rng):
    name, keys, position, gaussian, cached = np.random.get_state()
    return {'schema': 2, 'python': random.getstate(),
            'numpy': {'name': name, 'keys': keys.tolist(), 'position': position,
                      'has_gauss': gaussian, 'cached_gaussian': cached},
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'negative': negative_rng.getstate(), 'shuffle': shuffle_rng.get_state()}


def restore_rng(state, negative_rng, shuffle_rng):
    if state.get('schema') != 2:
        raise ValueError('Checkpoint lacks complete Python/NumPy RNG state; use a new run with this code')
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    if len(state['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('Checkpoint and runtime have different CUDA device counts')
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    negative_rng.setstate(state['negative'])
    shuffle_rng.set_state(state['shuffle'].cpu())


def assert_exact_state(expected, actual, path='state'):
    """Serialization audit, always exact, including floats. Device transfer is allowed."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(path + ': tensor dtype/shape mismatch')
        torch.testing.assert_close(expected.detach().cpu(), actual.detach().cpu(), rtol=0, atol=0,
                                   msg=lambda msg: path + ': ' + msg)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(path + ': mapping keys differ')
        for key in expected:
            assert_exact_state(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(path + ': sequence differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_exact_state(a, b, f'{path}[{i}]')
    elif expected != actual:
        raise AssertionError(f'{path}: {expected!r} != {actual!r}')


def restore_training_state(model, optimizer, saved, negative_rng, shuffle_rng):
    """Restore and audit before any further training computation or update."""
    from training.research_pipeline import load_model_state, trainable_state
    if saved.get('checkpoint_boundary') != 'epoch_end':
        raise ValueError('Only committed epoch-end last.pt checkpoints support resume')
    history = saved['history']
    if (not history or len(history) != saved['epoch']
            or [row['epoch'] for row in history] != list(range(1, saved['epoch'] + 1))
            or history[-1]['optimizer_steps'] != saved['optimizer_steps']):
        raise ValueError('Checkpoint epoch/update counters disagree with committed training history')
    if saved.get('scheduler_state') is not None or saved.get('grad_scaler_state') is not None:
        raise ValueError('This fixed-LR/full-precision protocol has no scheduler or gradient scaler')
    if saved['numerical_policy'] != numerical_policy():
        raise ValueError('Numerical/determinism policy changed since checkpoint; use matching runtime settings')
    load_model_state(model, saved['model_state'])
    optimizer.load_state_dict(saved['optimizer_state'])
    for name, module in model.named_modules():
        module.training = saved['module_modes'][name]
    assert_exact_state(saved['model_state'], trainable_state(model), 'restored_model')
    assert_exact_state(saved['optimizer_state'], optimizer.state_dict(), 'restored_optimizer')
    assert_exact_state(saved['module_modes'], {n: m.training for n, m in model.named_modules()}, 'module_modes')
    restore_rng(saved['rng'], negative_rng, shuffle_rng)
    assert_exact_state(saved['rng'], capture_rng(negative_rng, shuffle_rng), 'restored_rng')
    return {'model_and_buffers_exact': True, 'optimizer_exact': True, 'rng_exact': True,
            'module_modes_exact': True, 'scheduler': 'none (fixed learning rate)',
            'gradient_scaler': 'none (full precision)', 'boundary': 'epoch_end',
            'next_epoch': saved['epoch'] + 1, 'optimizer_steps': saved['optimizer_steps']}
