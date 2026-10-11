"""Shared RGAT study: six configurations, 1,000 entities, 30 total epochs, 5 warmup.

No preliminary training is performed. Each child initializes its own model;
only its own last.pt may resume it. Seed 0 remains the approved default.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

from training.experiment_io import digest, prepare_data, write_json
from training.process import run_logged
from training.research_pipeline import ExperimentConfig, run_experiment

from training.variants import SHARED_CONFIGURATIONS
from preprocessing.induced_subset import STRATEGY

PROTOCOL = 'shared_rgat_1000_30_5_seed0_v1'
SEEDS = (0,)
CONFIGURATIONS = SHARED_CONFIGURATIONS


def make_config(identifier, seed, *, raw_dir='data/raw/fb15k237',
                text_dir='data/text/fb15k237', text_batch_size=256, eval_batch_size=512):
    if identifier not in CONFIGURATIONS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('The full study requires a shared configuration and a nonnegative 32-bit seed')
    architecture, warmup = identifier, 5
    cfg = ExperimentConfig(variant=architecture, max_entities=1000, epochs=30,
        warmup=warmup, seed=seed, subset_seed=0, lr=.001, num_negatives=4,
        train_batch_size=32768, text_batch_size=text_batch_size, max_text_length=64,
        eval_every=5, eval_batch_size=eval_batch_size, raw_dir=str(raw_dir), text_dir=str(text_dir),
        selection_policy='scheduled_only', record_step_losses=True)
    cfg.validate()
    return cfg


def expected_budget(identifier, train_triples):
    if identifier not in CONFIGURATIONS:
        raise ValueError('Unknown shared configuration')
    import math
    steps = math.ceil(train_triples / 32768)
    return {'steps_per_epoch': steps, 'optimizer_steps': 30 * steps,
            'structural_optimizer_steps': 5 * steps,
            'integrated_optimizer_steps': 25 * steps, 'baseline_optimizer_steps': 0}


def verify_manifest(manifest):
    stats = manifest['statistics']
    if (stats['entities'] != 1000 or stats['relations'] != 237
            or manifest['max_entities'] != 1000 or manifest['subset_seed'] != 0
            or manifest['selection_strategy'] != STRATEGY
            or any(stats[k] < 1 for k in ('train_triples', 'valid_triples', 'test_triples'))):
        raise ValueError('Shared study requires the same train-induced 1,000-entity subset, subset seed 0')


def build_commands(args, root, manifest):
    runner = Path(__file__).resolve().parents[1] / 'run_shared_rgat_comparison.py'
    commands = []
    for identifier in CONFIGURATIONS:
        for seed in (getattr(args, 'seed', 0),):
            directory = root / f'{identifier}_seed{seed}'
            command = [sys.executable, '-u', str(runner), '--single', identifier, '--seed', str(seed),
                       '--manifest', str(manifest), '--output-dir', str(directory)]
            for key in ('raw_dir', 'text_dir', 'cache_dir', 'text_batch_size', 'eval_batch_size'):
                command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
            for key in ('skip_completed', 'resume'):
                if getattr(args, key):
                    command.append('--' + key.replace('_', '-'))
            commands.append((identifier, seed, directory, command))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--raw-dir', default='data/raw/fb15k237')
    parser.add_argument('--text-dir', default='data/text/fb15k237')
    parser.add_argument('--cache-dir', default='data/cache/pooled_text')
    parser.add_argument('--text-batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=512)
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--single', choices=CONFIGURATIONS, help='One independent child; also useful for explicit manual launch')
    parser.add_argument('--seed', type=int, default=0, help='Independent study seed; approved default: 0. Use a fresh output directory for another seed.')
    parser.add_argument('--manifest')
    parser.add_argument('--report-only', action='store_true', help='Regenerate reports from all six completed results; no training')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be a nonnegative 32-bit integer')
    root = Path(args.output_dir)
    from training.shared_reports import shared_report
    if args.report_only:
        if args.single:
            parser.error('--report-only requires a complete study directory')
        results = [json.loads((root / f'{c}_seed{s}' / 'results.json').read_text(encoding='utf-8'))
                   for c in CONFIGURATIONS for s in (args.seed,)]
        shared_report(root, results, seed=args.seed)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Shared research training requires CUDA; no CPU fallback or smoke training was started')
    cfg_kwargs = dict(raw_dir=args.raw_dir, text_dir=args.text_dir,
                      text_batch_size=args.text_batch_size, eval_batch_size=args.eval_batch_size)
    if args.single:
        run_experiment(make_config(args.single, args.seed, **cfg_kwargs), root,
                       manifest_path=args.manifest, cache_dir=args.cache_dir,
                       skip_completed=args.skip_completed, resume=args.resume)
        return
    manifest_path = root / 'dataset_manifest.json'
    configs = {c: asdict(make_config(c, args.seed, **cfg_kwargs)) for c in CONFIGURATIONS}
    config = {'protocol': PROTOCOL if args.seed == 0 else PROTOCOL.replace('seed0', f'seed{args.seed}'), 'seeds': [args.seed], 'configurations': configs,
              'cache_dir': str(args.cache_dir)}
    config_path = root / 'comparison_config.json'
    if config_path.exists():
        if digest(json.loads(config_path.read_text(encoding='utf-8'))) != digest(config):
            raise ValueError('Study settings changed; use a new run tag')
        if not (args.resume or args.skip_completed):
            raise FileExistsError('Study exists; explicitly resume/skip completed or choose a new directory')
    elif root.exists() and any(root.iterdir()):
        if {p.name for p in root.iterdir()} != {'dataset_manifest.json'}:
            raise FileExistsError('Refusing an unrecognized nonempty output directory')
    _, _, _, manifest = prepare_data(args.raw_dir, 1000, 0, manifest_path)
    verify_manifest(manifest)
    write_json(config_path, config)
    print('Shared subset:', manifest['statistics'], '\nManifest:', manifest['sha256'], flush=True)
    results = []
    for identifier, seed, directory, command in build_commands(args, root, manifest_path):
        print(f'Run {len(results)+1}/6: {identifier}; seed={seed}; {expected_budget(identifier, manifest['statistics']['train_triples'])}', flush=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_logged(command, root / 'logs' / f'{identifier}_seed{seed}_{stamp}.log')
        results.append(json.loads((directory / 'results.json').read_text(encoding='utf-8')))
        write_json(root / 'runs.json', results)
    summary = shared_report(root, results, seed=args.seed)
    print('Selected by mean validation MRR:', summary['selected_by_validation'], flush=True)
    print('Reports:', root, flush=True)

