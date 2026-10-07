"""Run matched variants and initialization seeds on one fixed larger subset."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='experiments/checkpoints/kg_only_baseline_rgat_complex.pt')
    parser.add_argument('--max-entities', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--output-dir', default='experiments/phase5_comparisons')
    args = parser.parse_args()
    root = Path(args.output_dir)
    variants = ('original', 'residual', 'no-refinement')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new comparison --output-dir')
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in variants:
        for seed in args.seeds:
            directory = root / f'{variant}_seed{seed}'
            print('Running', variant, 'seed', seed, flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('run_phase5_training.py')),
                            '--checkpoint', args.checkpoint, '--variant', variant,
                            '--max-entities', str(args.max_entities), '--epochs', str(args.epochs),
                            '--seed', str(seed), '--subset-seed', str(args.subset_seed),
                            '--output-dir', str(directory)], check=True)
            result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
            records.append({'variant': variant, 'seed': seed, 'best_epoch': result['best_epoch'],
                            'validation': result['full_model_validation'],
                            'test': result['full_model_test'],
                            'baseline_validation': result['settings']['baseline_validation'],
                            'baseline_test': result['settings']['baseline_test']})
            (root / 'runs.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    summary = {}
    for variant in variants:
        runs = [r for r in records if r['variant'] == variant]
        summary[variant] = {}
        for split in ('validation', 'test'):
            summary[variant][split] = {}
            for metric in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10'):
                values = [r[split][metric] for r in runs]
                summary[variant][split][metric] = {'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0}
    # Select variant by validation only; test results are descriptive.
    winner = max(summary, key=lambda v: summary[v]['validation']['MRR']['mean'])
    report = {'settings': vars(args), 'variants': summary, 'selected_by_validation': winner,
              'note': 'Restricted subset candidates. Seeds vary new model/training initialization; all reuse the same seed-0 pretrained structural checkpoint.'}
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
