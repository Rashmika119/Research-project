"""Final full FB15k-237 study: residual versus Shared C, seed 0, 100 epochs.

No preliminary training is performed. Each child initializes its own model;
only its own last.pt may resume it. Only seed 0 is approved for this study.
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

PROTOCOL = 'full_fb15k237_residual_vs_shared_c_100_10_v1'
SEEDS = (0,)
CONFIGURATIONS = {
    'residual-no-softprompt-warmup': ('residual-no-softprompt', 10),
    'shared-c-gated': ('shared-c-gated', 10),
    'shared-c-ungated': ('shared-c-ungated', 10),
}
COUNTS = (14541, 237, 272115, 17535, 20466)


def make_config(identifier, seed, *, raw_dir='data/raw/fb15k237',
                text_dir='data/text/fb15k237', text_batch_size=256, eval_batch_size=512):
    if identifier not in CONFIGURATIONS or type(seed) is not int or seed != 0:
        raise ValueError('The full study requires a selected configuration and seed 0')
    if (text_batch_size, eval_batch_size) != (256, 512):
        raise ValueError('Final protocol fixes text/evaluation batch sizes at 256/512')
    from dataclasses import replace
    from training.full_study import make_config as original_config
    cfg = replace(original_config('residual-no-softprompt-warmup', seed,
                  raw_dir=raw_dir, text_dir=text_dir), variant=CONFIGURATIONS[identifier][0])
    cfg.validate()
    return cfg


def validate_seeds(seeds):
    values = tuple(seeds)
    if values != SEEDS or any(type(s) is not int for s in values):
        raise ValueError('The final study uses only seed 0')
    return values



def expected_budget(identifier):
    if identifier not in CONFIGURATIONS:
        raise ValueError('Unknown final configuration')
    return {'steps_per_epoch': 9, 'optimizer_steps': 900,
            'structural_optimizer_steps': 90, 'baseline_optimizer_steps': 0,
            'integrated_optimizer_steps': 810}


# Full mappings/splits/graph preparation remains in the common data implementation.
from training.full_study import verify_manifest


def preflight(root, configs, manifest):
    """Read saved identities/hashes without constructing models or loading tensors."""
    from training.experiment_io import environment, file_digest
    verify_manifest(manifest)
    current = None
    for name, cfg in configs.items():
        directory = Path(root) / name
        path = directory / 'config.json'
        if not path.exists():
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f'Unrecognized run directory: {directory}')
            continue
        saved = json.loads(path.read_text(encoding='utf-8'))
        if current is None:
            current = environment()
        if saved['config'] != asdict(cfg) or saved['manifest_sha256'] != manifest['sha256']:
            raise ValueError(f'Saved settings/dataset differ: {directory}; use matching source or a new tag')
        if any(saved[k] != current[k] for k in ('source_sha256', 'packages', 'python', 'cuda', 'device')):
            raise ValueError(f'Saved source/runtime differs: {directory}; restore its matching environment')
        result_path = directory / 'results.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result['run_id'] != saved['run_id'] or result['status'] != 'complete':
                raise ValueError(f'Invalid completion marker: {directory}')
            for checkpoint, checksum in result['checkpoint_sha256'].items():
                if file_digest(directory/checkpoint) != checksum:
                    raise ValueError(f'Changed checkpoint: {directory/checkpoint}')
        print('Existing run identity verified:', directory, flush=True)



def build_commands(args, root, manifest):
    runner = Path(__file__).resolve().parents[1] / 'run_final_comparison.py'
    commands = []
    for identifier in CONFIGURATIONS:
        for seed in args.seeds:
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
    parser.add_argument('--seed', type=int, help='Seed for an explicit --single child')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS), help='Study seed; only 0 is supported.')
    parser.add_argument('--manifest')
    parser.add_argument('--report-only', action='store_true', help='Regenerate reports from all requested completed configuration/seed results; no training')
    args = parser.parse_args()
    args.seeds = validate_seeds(args.seeds)
    if args.single and (args.seed != 0):
        parser.error('--single requires --seed 0')
    if not args.single and args.seed is not None:
        parser.error('Use --seeds for a study, --seed only with --single')
    root = Path(args.output_dir)
    from training.final_reports import final_report
    if args.report_only:
        if args.single:
            parser.error('--report-only requires a complete study directory')
        results = [json.loads((root / f'{c}_seed{s}' / 'results.json').read_text(encoding='utf-8'))
                   for c in CONFIGURATIONS for s in args.seeds]
        final_report(root, results, seeds=args.seeds)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Full research training requires CUDA; no CPU fallback or smoke training was started')
    cfg_kwargs = dict(raw_dir=args.raw_dir, text_dir=args.text_dir,
                      text_batch_size=args.text_batch_size, eval_batch_size=args.eval_batch_size)
    if args.single:
        run_experiment(make_config(args.single, args.seed, **cfg_kwargs), root,
                       manifest_path=args.manifest, cache_dir=args.cache_dir,
                       skip_completed=args.skip_completed, resume=args.resume)
        return
    manifest_path = root / 'dataset_manifest.json'
    configs = {f'{c}_seed{s}': asdict(make_config(c, s, **cfg_kwargs)) for c in CONFIGURATIONS for s in args.seeds}
    config = {'protocol': PROTOCOL, 'seeds': list(args.seeds), 'configurations': configs,
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
    _, _, _, manifest = prepare_data(args.raw_dir, 0, 0, manifest_path)
    verify_manifest(manifest)
    preflight(root, {name: ExperimentConfig(**cfg) for name, cfg in configs.items()}, manifest)
    write_json(config_path, config)
    print('Full dataset:', manifest['statistics'], '\nManifest:', manifest['sha256'], flush=True)
    results = []
    for identifier, seed, directory, command in build_commands(args, root, manifest_path):
        print(f'Run {len(results)+1}/{len(CONFIGURATIONS)*len(args.seeds)}: {identifier}; seed={seed}; {expected_budget(identifier)}', flush=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_logged(command, root / 'logs' / f'{identifier}_seed{seed}_{stamp}.log')
        results.append(json.loads((directory / 'results.json').read_text(encoding='utf-8')))
        write_json(root / 'runs.json', results)
    summary = final_report(root, results, seeds=args.seeds)
    print('Selected by mean validation MRR:', summary['selected_by_validation'], flush=True)
    print('Reports:', root, flush=True)

