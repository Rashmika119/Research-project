"""Sequential, restartable four-configuration warmup ablation (12 default runs)."""
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

from training.experiment_io import digest, prepare_data, write_json
from training.ablation_reports import ablation_report
from training.process import run_logged
from training.variants import CONFIGURATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-entities', type=int, default=5000)
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
    parser.add_argument('--output-dir', default='experiments/research/warmup_ablation_5000')
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.max_entities < 2:
        raise ValueError('Use unique seeds and a subset of at least two entities')
    if not 0 < args.warmup_epochs < args.epochs:
        raise ValueError('Warmup ablation needs 0 < warmup < total epochs')
    root = Path(args.output_dir)
    config = {k: v for k, v in vars(args).items() if k not in ('skip_completed', 'resume', 'output_dir')}
    config.update(configurations=list(CONFIGURATIONS), protocol='warmup_ablation_induced_v1',
                  lr=0.001, train_batch_size=0, eval_every=1, num_negatives=4)
    config_path = root / 'comparison_config.json'
    if config_path.exists():
        if digest(json.loads(config_path.read_text(encoding='utf-8'))) != digest(config):
            raise ValueError('Comparison settings changed; choose a new directory')
        if not args.skip_completed and not args.resume:
            raise FileExistsError('Comparison exists; use --skip-completed and/or --resume')
    elif root.exists() and any(root.iterdir()):
        if {p.name for p in root.iterdir()} != {'subset_manifest.json'}:
            raise FileExistsError('Unrecognized nonempty comparison directory')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'subset_manifest.json'
    _, _, _, prepared = prepare_data(args.raw_dir, args.max_entities, args.subset_seed, manifest)
    print('Subset statistics:', json.dumps(prepared['statistics']), flush=True)
    print('Shared manifest SHA256:', prepared['sha256'], flush=True)
    write_json(config_path, config)
    runner = Path(__file__).resolve().parents[1] / 'run_phase5_training.py'
    results = []
    for identifier, (architecture, enabled) in CONFIGURATIONS.items():
        for seed in args.seeds:
            directory = root / f'{identifier}_seed{seed}'
            command = [sys.executable, str(runner), '--configuration', identifier, '--seed', str(seed),
                       '--manifest', str(manifest), '--output-dir', str(directory),
                       '--lr', '0.001', '--num-negatives', '4', '--train-batch-size', '0', '--eval-every', '1']
            for key in ('max_entities', 'epochs', 'warmup_epochs', 'subset_seed', 'text_batch_size',
                        'eval_batch_size', 'max_text_length', 'lm_name', 'raw_dir', 'text_dir', 'cache_dir'):
                command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
            if args.skip_completed:
                command.append('--skip-completed')
            if args.resume:
                command.append('--resume')
            print(f'Run {len(results) + 1}/{len(CONFIGURATIONS) * len(args.seeds)}: '
                  f'{identifier}; architecture={architecture}; warmup={enabled}; seed={seed}', flush=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            run_logged(command, root / 'logs' / f'{identifier}_seed{seed}_{stamp}.log')
            results.append(json.loads((directory / 'results.json').read_text(encoding='utf-8')))
            write_json(root / 'runs.json', results)
    summary = ablation_report(root, results, args.seeds)
    print(json.dumps(summary, indent=2), flush=True)
