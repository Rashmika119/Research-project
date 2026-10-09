"""CSV/JSON reports and plots; architecture selection uses validation only."""
import csv
import json
import statistics
from pathlib import Path

from training.experiment_io import digest, write_json
from training.variants import VARIANTS


METRICS = ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_run(directory, result):
    directory = Path(directory)
    rows = []
    for entry in result['history']:
        row = {k: v for k, v in entry.items() if k != 'validation'}
        row.update({'val_' + k: v for k, v in entry.get('validation', {}).items()})
        rows.append(row)
    write_csv(directory / 'history.csv', rows)
    write_csv(directory / 'metrics.csv', [{'split': split, **result[split]} for split in ('validation', 'test')])


def summarize(results, seeds=(0, 1, 2), variants=VARIANTS):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Seeds must be nonempty and unique')
    expected = {(v, s) for v in variants for s in seeds}
    actual = [(r['settings']['variant'], r['settings']['seed']) for r in results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing, duplicate, or unexpected variant/seed runs')
    manifests = {r['manifest_sha256'] for r in results}
    if len(manifests) != 1:
        raise ValueError('Cannot aggregate different datasets/subsets')
    protocols = []
    for result in results:
        if result['status'] != 'complete' or result['initialization'] != 'scratch':
            raise ValueError('Only completed scratch runs may enter this comparison')
        cfg = {k: v for k, v in result['settings'].items() if k not in ('variant', 'seed')}
        protocols.append(digest(cfg))
        if result['ranking_policy'] != 'average_exact_ties':
            raise ValueError('Mixed ranking policies')
    if len(set(protocols)) != 1:
        raise ValueError('Run optimization settings differ')
    if len({digest(r.get('lm_identity')) for r in results}) != 1:
        raise ValueError('Frozen LM identities differ across runs')
    if len({r.get('environment', {}).get('source_sha256') for r in results}) != 1:
        raise ValueError('Source versions differ across runs')
    # Require paired structural initialization as well as identical input facts.
    for seed in seeds:
        if len({r['initial_structural_sha256'] for r in results if r['settings']['seed'] == seed}) != 1:
            raise ValueError('Variants did not start with identical structural weights for this seed')
    summaries, table = {}, []
    for variant in variants:
        runs = [r for r in results if r['settings']['variant'] == variant]
        summary = {'best_epochs': [r['best_epoch'] for r in runs],
                   'optimizer_steps': [r['optimizer_steps'] for r in runs]}
        flat = {'variant': variant, 'seeds': ','.join(str(s) for s in seeds)}
        for split in ('validation', 'test'):
            summary[split] = {}
            for metric in METRICS:
                values = [r[split][metric] for r in runs]
                entry = {'mean': statistics.mean(values),
                         'std': statistics.stdev(values) if len(values) > 1 else None}
                summary[split][metric] = entry
                flat[f'{split}_{metric}_mean'] = entry['mean']
                flat[f'{split}_{metric}_std'] = entry['std']
        summary['duration_seconds_mean'] = statistics.mean(r['duration_seconds'] for r in runs)
        peaks = [r['peak_gpu_memory_bytes'] for r in runs if r['peak_gpu_memory_bytes'] is not None]
        summary['peak_gpu_memory_bytes_max'] = max(peaks) if peaks else None
        flat.update(best_epochs=','.join(str(r['best_epoch']) for r in runs),
                    duration_seconds_mean=summary['duration_seconds_mean'],
                    peak_gpu_memory_bytes_max=summary['peak_gpu_memory_bytes_max'])
        table.append(flat)
        summaries[variant] = summary
    winner = max(variants, key=lambda v: summaries[v]['validation']['MRR']['mean'])
    return {'variants': summaries, 'selected_by_validation': winner,
            'selection_metric': 'mean validation MRR', 'seeds': list(seeds),
            'manifest_sha256': next(iter(manifests)),
            'tie_break': 'First variant in configured order for exactly equal means',
            'scope': 'Pilot subset candidates; not comparable to full-dataset MRR'}, table


def comparison_report(directory, results, seeds=(0, 1, 2), variants=VARIANTS):
    directory = Path(directory)
    summary, table = summarize(results, seeds, variants)
    write_json(directory / 'runs.json', results)
    write_json(directory / 'summary.json', summary)
    write_csv(directory / 'comparison.csv', table)
    individual = []
    for r in results:
        row = {'variant': r['settings']['variant'], 'seed': r['settings']['seed'],
               'best_epoch': r['best_epoch'], 'warmup_epochs': r['warmup_epochs'],
               'main_epochs': r['main_epochs'], 'optimizer_steps': r['optimizer_steps'],
               'duration_seconds': r['duration_seconds'],
               'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes']}
        row.update({f'{split}_{m}': r[split][m] for split in ('validation', 'test') for m in METRICS})
        individual.append(row)
    write_csv(directory / 'runs.csv', individual)
    plot_comparison(directory, results, summary, variants)
    return summary


def plot_comparison(directory, results, summary, variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(variants), 2, figsize=(12, 3 * len(variants)), squeeze=False)
    for i, variant in enumerate(variants):
        for result in (r for r in results if r['settings']['variant'] == variant):
            label = f"seed {result['settings']['seed']}"
            history = result['history']
            axes[i, 0].plot([h['epoch'] for h in history], [h['loss'] for h in history], label=label)
            main = [h for h in history if h['phase'] == 'main' and 'validation' in h]
            axes[i, 1].plot([h['epoch'] for h in main], [h['validation']['MRR'] for h in main], label=label)
        for j, title in enumerate(('BCE loss (warmup included)', 'Integrated validation MRR')):
            axes[i, j].set(title=f'{variant}: {title}', xlabel='Total training epoch')
            axes[i, j].legend()
            axes[i, j].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(directory) / 'training_curves.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [summary['variants'][v]['validation']['MRR'] for v in variants]
    ax.bar(variants, [v['mean'] for v in values], yerr=[v['std'] or 0 for v in values], capsize=4)
    ax.set(ylabel='Mean validation MRR ± sample SD', title='Pilot architecture selection')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(Path(directory) / 'validation_comparison.png', dpi=150)
    plt.close(fig)


def final_comparison(baseline, selected, output_dir):
    if baseline['dataset_scope'] != 'full' or selected['dataset_scope'] != 'full':
        raise ValueError('Final comparison requires two FULL-dataset experiments')
    if baseline['dataset_sha256'] != selected['dataset_sha256']:
        raise ValueError('Final comparison dataset versions differ')
    if baseline['ranking_policy'] != selected['ranking_policy']:
        raise ValueError('Final comparison ranking policies differ')
    if baseline['settings']['variant'] != 'baseline' or selected['settings']['variant'] not in VARIANTS:
        raise ValueError('Expected independent graph baseline and a selected integrated architecture')
    if baseline['initialization'] != 'scratch' or selected['initialization'] != 'scratch':
        raise ValueError('Final comparison requires independent scratch experiments')
    rows = [{'model': label, 'variant': r['settings']['variant'], 'best_epoch': r['best_epoch'],
             'training_epochs': r['settings']['epochs'], 'optimizer_steps': r['optimizer_steps'],
             **{f'test_{m}': r['test'][m] for m in METRICS}}
            for label, r in [('independent_graph_baseline', baseline), ('selected_full_model', selected)]]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'final_comparison.json', {'results': rows,
        'note': 'Same full dataset and evaluator. Report differing optimizer-step budgets; '
                'pilot scores and checkpoint weights are not reused.'})
    write_csv(directory / 'final_comparison.csv', rows)
    return rows
