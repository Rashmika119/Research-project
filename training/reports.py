"""CSV/JSON reports and plots; architecture selection uses validation only."""
import csv
import statistics as stats
from pathlib import Path



METRICS = ('MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Optimistic_MRR',
           'Tie_query_fraction', 'Mean_tied_candidates', 'Max_tied_candidates')


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




def mean_sd(values):
    return {'mean': stats.mean(values), 'std': stats.stdev(values) if len(values) > 1 else None}


def loss_behavior(rows):
    if not rows:
        return None
    losses = [r['loss'] for r in rows]
    differences = [b - a for a, b in zip(losses, losses[1:])]
    median = stats.median(differences) if differences else 0.
    mad = stats.median(abs(d - median) for d in differences) if differences else 0.
    threshold = max(median + 3 * mad, 1e-8)
    epochs = [r['epoch'] for r in rows]
    mean_x, mean_y = stats.mean(epochs), stats.mean(losses)
    denominator = sum((x - mean_x) ** 2 for x in epochs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(epochs, losses)) / denominator if denominator else None
    improvement = losses[0] - min(losses)
    target = losses[0] - .95 * improvement
    return {'initial_loss': losses[0], 'final_loss': losses[-1], 'minimum_loss': min(losses),
            'linear_loss_slope_per_epoch': slope,
            'loss_change_std': stats.stdev(differences) if len(differences) > 1 else None,
            'upward_steps': sum(d > 0 for d in differences),
            'spike_threshold': threshold,
            'spike_epochs': [rows[i + 1]['epoch'] for i, d in enumerate(differences) if d > threshold],
            'epoch_at_95pct_observed_loss_reduction': next((r['epoch'] for r in rows if r['loss'] <= target), None)
                if improvement > 0 else None}


def behavior(r):
    histories = r['history']
    main = [h for h in histories if h['phase'] == 'main']
    validation = [h for h in main if 'validation' in h]
    # Compare the same epochs for all models, independently of warmup eligibility.
    matched = [h for h in validation if h['epoch'] >= 15]
    deltas = [b['validation']['MRR'] - a['validation']['MRR'] for a, b in zip(matched, matched[1:])]
    tail = validation[-6:]
    return {'configuration': r['configuration'], 'seed': r['settings']['seed'],
            'warmup_loss': loss_behavior([h for h in histories if h['phase'] == 'structural_warmup']),
            'main_loss': loss_behavior(main), 'main_model': 'baseline' if r['architecture'] == 'baseline' else 'integrated',
            'best_epoch': r['best_epoch'], 'best_before_final': r['best_epoch'] < 100,
            'best_to_final_validation_mrr_drop': r['validation']['MRR'] - r['final_epoch_validation']['MRR'],
            'loss_falls_while_validation_stalls_last_6_evaluations':
                len(tail) > 1 and tail[-1]['loss'] < tail[0]['loss'] and
                max(h['validation']['MRR'] for h in tail[1:]) <= tail[0]['validation']['MRR'] + 1e-6,
            'matched_epochs_15_100_mrr_delta_sd': stats.stdev(deltas),
            'matched_epochs_15_100_ranking_declines': sum(x < 0 for x in deltas),
            'first_epoch_at_95pct_observed_best_main_mrr': next(h['epoch'] for h in validation
                if h['validation']['MRR'] >= .95 * r['validation']['MRR']),
            'severe_tie_evaluations': [h['epoch'] for h in histories if 'validation' in h and
                (h['validation']['Tie_query_fraction'] >= .5 or h['validation']['Max_tied_candidates'] >= 1454.1)]}
