"""Sequential, restartable five-variant pilot orchestration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.experiment_io import digest, prepare_data, write_json
from training.reports import comparison_report
from training.variants import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-entities', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--text-batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--max-text-length', type=int, default=64)
    parser.add_argument('--lm-name', default='roberta-base')
    parser.add_argument('--raw-dir', default='data/raw/fb15k237')
    parser.add_argument('--text-dir', default='data/text/fb15k237')
    parser.add_argument('--cache-dir', default='data/cache/pooled_text')
    parser.add_argument('--output-dir', default='experiments/research/pilot')
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.max_entities < 2:
        raise ValueError('Use unique seeds and a subset of at least two entities')
    if not 0 < args.warmup_epochs < args.epochs:
        raise ValueError('The five-way warmup ablation needs 0 < warmup < total epochs')
    root = Path(args.output_dir)
    config = {k: v for k, v in vars(args).items() if k not in ('skip_completed', 'resume', 'output_dir')}
    config['variants'] = list(VARIANTS)
    config_path = root / 'comparison_config.json'
    if config_path.exists():
        if digest(json.loads(config_path.read_text(encoding='utf-8'))) != digest(config):
            raise ValueError('Comparison settings changed; choose a new directory')
        if not args.skip_completed and not args.resume:
            raise FileExistsError('Comparison exists; use --skip-completed and/or --resume')
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError('Unrecognized nonempty comparison directory')
    root.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    manifest = root / 'subset_manifest.json'
    prepare_data(args.raw_dir, args.max_entities, args.subset_seed, manifest)
    runner = Path(__file__).resolve().parents[1] / 'run_phase5_training.py'
    results = []
    for variant in VARIANTS:
        for seed in args.seeds:
            directory = root / f'{variant}_seed{seed}'
            command = [sys.executable, str(runner), '--variant', variant, '--seed', str(seed),
                       '--manifest', str(manifest), '--output-dir', str(directory)]
            for key in ('max_entities', 'epochs', 'warmup_epochs', 'subset_seed', 'text_batch_size',
                        'eval_batch_size', 'max_text_length', 'lm_name', 'raw_dir', 'text_dir', 'cache_dir'):
                command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
            if args.skip_completed:
                command.append('--skip-completed')
            if args.resume:
                command.append('--resume')
            print(f'Running {variant}, seed {seed} ({len(results) + 1}/{len(VARIANTS) * len(args.seeds)})', flush=True)
            subprocess.run(command, check=True)
            results.append(json.loads((directory / 'results.json').read_text(encoding='utf-8')))
            write_json(root / 'runs.json', results)
    summary = comparison_report(root, results, args.seeds)
    print(json.dumps(summary, indent=2), flush=True)
