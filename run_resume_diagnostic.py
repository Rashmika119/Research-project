"""Compare two uninterrupted controls with a checkpoint-resumed tiny experiment.

Uses real RGAT/ComplEx; --real-lm additionally uses verified pretrained RoBERTa.
Synthetic facts and short diagnostic budgets are NOT research benchmarks.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def tensor_differences(a, b, path='state'):
    import torch
    rows = []
    if isinstance(a, torch.Tensor):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return [{'path': path, 'error': 'shape/dtype mismatch'}]
        delta = (a.double() - b.double()).abs()
        rows.append({'path': path, 'dtype': str(a.dtype), 'mismatched': int((a != b).sum()),
                     'elements': a.numel(), 'max_abs': delta.max().item() if a.numel() else 0.,
                     'max_rel': (delta / b.double().abs().clamp_min(1e-30)).max().item() if a.numel() else 0.})
    elif isinstance(a, dict):
        for key in a.keys() & b.keys():
            rows.extend(tensor_differences(a[key], b[key], f'{path}.{key}'))
    elif isinstance(a, (tuple, list)):
        for i, (left, right) in enumerate(zip(a, b)):
            rows.extend(tensor_differences(left, right, f'{path}[{i}]'))
    return rows


def compare(left_dir, right_dir, report_path):
    import torch
    from training.reproducibility import assert_exact_state
    fields = ('model_state', 'optimizer_state', 'rng', 'epoch', 'optimizer_steps',
              'best_epoch', 'best_val_mrr', 'best_model_state', 'module_modes',
              'initial_validation', 'numerical_policy', 'scheduler_state', 'grad_scaler_state')
    left, right = [torch.load(folder / 'last.pt', weights_only=True, map_location='cpu')
                   for folder in (left_dir, right_dir)]
    a, b = ({key: value[key] for key in fields} for value in (left, right))
    for state, folder in ((a, left_dir), (b, right_dir)):
        result = json.loads((folder / 'results.json').read_text())
        state['validation'], state['test'] = result['validation'], result['test']
        state['history'] = [{k: v for k, v in row.items() if k != 'epoch_seconds'} for row in result['history']]
    report = {'left': str(left_dir), 'right': str(right_dir), 'rtol': 0, 'atol': 0,
              'tensors': tensor_differences(a, b), 'left_history': a['history'], 'right_history': b['history'],
              'left_metrics': {'validation': a['validation'], 'test': a['test']},
              'right_metrics': {'validation': b['validation'], 'test': b['test']}}
    try:
        assert_exact_state(a, b)
        report['status'] = 'passed_exactly'
    except AssertionError as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return {'status': report['status'], 'report': str(report_path),
            'max_tensor_abs_difference': max((r.get('max_abs', float('inf')) for r in report['tensors']), default=0.)}


def run(args):
    import torch
    from dataclasses import replace
    from run_pretrained_check import select_device
    from tests.test_research_pipeline import tiny_data, small_config, make_tokens, fake_lm
    from training import research_pipeline as pipeline
    from training.reproducibility import numerical_policy
    device = select_device(args.device)
    torch.set_num_threads(1)
    data = tiny_data()
    cfg = small_config(variant=args.variant, train_batch_size=12)
    if args.real_lm:
        from models.pretrained import load_pretrained_tokenizer
        tokenizer = load_pretrained_tokenizer()
        cfg = replace(cfg, lm_name='roberta-base', text_batch_size=2,
                      model={**cfg.model, 'dim': 32})
        def tokens(n):
            encoded = tokenizer([f'Example item {i} has a description.' for i in range(n)],
                                padding=True, truncation=True, max_length=16, return_tensors='pt')
            return {k: encoded[k] for k in ('input_ids', 'attention_mask')}
        fixture = nullcontext()
    else:
        tokens = make_tokens
        fixture = patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    inputs = tokens(data.num_entities), tokens(data.num_relations)
    report = {'device': str(device), 'real_pretrained_lm': args.real_lm,
              'variant': args.variant, 'repeats': [], 'status': 'running'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with fixture:
            for repeat in range(args.repeats):
                root = args.output_dir / f'repeat_{repeat}'
                if root.exists():
                    raise FileExistsError(f'Use a fresh diagnostic output directory: {root}')
                clean, control, resumed = [root / name for name in ('clean', 'control', 'resumed')]
                # Real-LM single-repeat checks must restore integrated Adam
                # moments, not only interrupt before the LM was first trained through.
                interrupt_epoch = 2 if args.real_lm else 1 + repeat % 2
                run_cfg = replace(cfg, seed=repeat)
                pipeline.run_experiment(run_cfg, clean, full_dataset=data, tokens=inputs)
                pipeline.run_experiment(run_cfg, control, full_dataset=data, tokens=inputs)
                real_save = pipeline.save_checkpoint
                def interrupt(path, value):
                    real_save(path, value)
                    if Path(path).name == 'last.pt' and value['epoch'] == interrupt_epoch:
                        raise InterruptedError('Intentional disconnect after atomic epoch commit')
                with patch.object(pipeline, 'save_checkpoint', side_effect=interrupt):
                    try:
                        pipeline.run_experiment(run_cfg, resumed, full_dataset=data, tokens=inputs)
                    except InterruptedError:
                        pass
                    else:
                        raise AssertionError('Interruption hook was not reached')
                pipeline.run_experiment(run_cfg, resumed, full_dataset=data, tokens=inputs, resume=True)
                # Always write both diagnostics, even when one comparison fails.
                comparisons, errors = {}, []
                for name, target in (('uninterrupted_control', control), ('checkpoint_resume', resumed)):
                    try:
                        comparisons[name] = compare(clean, target, root / f'{name}.json')
                    except AssertionError as error:
                        errors.append(f'{name}: {error}')
                report['repeats'].append({'seed': repeat, 'interrupted_after_epoch': interrupt_epoch,
                                           'comparisons': comparisons, 'errors': errors})
                if errors:
                    raise AssertionError('\n'.join(errors))
        report['status'] = 'passed_exactly'
        print('Resume diagnostic PASSED exactly on', device, flush=True)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['numerical_policy'] = numerical_policy()
        (args.output_dir / 'resume_diagnostic.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--real-lm', action='store_true')
    parser.add_argument('--variant', choices=('original', 'residual', 'no-refinement',
                                             'original-no-warmup', 'residual-no-softprompt'), default='original')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run(args)
