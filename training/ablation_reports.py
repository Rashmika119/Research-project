"""Descriptive paired ablation analysis; validation alone selects the configuration."""
import math
from pathlib import Path
import statistics as stats

from training.experiment_io import digest, write_json
from training.reports import write_csv
from training.variants import CONFIGURATIONS, configuration_id

METRICS = ('MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Optimistic_MRR',
           'Tie_query_fraction', 'Mean_tied_candidates', 'Max_tied_candidates')
PLOTS = ('mean_training_loss.png', 'mean_validation_mrr.png', 'mean_validation_hits10.png',
         'test_mrr.png', 'test_hits10.png', 'per_seed_loss.png', 'per_seed_validation_mrr.png',
         'training_time.png')


def mean_sd(values):
    return {'mean': stats.mean(values), 'std': stats.stdev(values) if len(values) > 1 else None}


def validate_runs(results, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    actual = [(r['configuration'], r['settings']['seed']) for r in results]
    if len(actual) != len(set(actual)) or set(actual) != {(c, s) for c in CONFIGURATIONS for s in seeds}:
        raise ValueError('Missing, duplicate or unexpected configuration/seed runs')
    for field in ('manifest_sha256', 'dataset_sha256', 'candidate_entities', 'ranking_policy'):
        if len({r[field] for r in results}) != 1:
            raise ValueError('Mixed comparison field: ' + field)
    for field in ('lm_identity', 'environment', 'numerical_policy'):
        if len({digest(r.get(field)) for r in results}) != 1:
            raise ValueError('Mixed model/source/environment/numerical identity: ' + field)
    protocols = set()
    enabled_warmups = set()
    for r in results:
        cfg = r['settings']
        architecture, enabled = CONFIGURATIONS[r['configuration']]
        if (r['status'] != 'complete' or r['initialization'] != 'scratch'
                or r.get('baseline_checkpoint_used', False) or r['ranking_policy'] != 'average_exact_ties'):
            raise ValueError('Only completed scratch runs with corrected ranking may be compared')
        if (cfg['variant'] != architecture or configuration_id(architecture, cfg['warmup']) != r['configuration']
                or r['warmup_epochs'] != cfg['warmup'] or bool(cfg['warmup']) != enabled):
            raise ValueError('Configuration architecture/warmup mismatch')
        if enabled:
            enabled_warmups.add(cfg['warmup'])
        if (cfg['train_batch_size'] != 0 or r['steps_per_epoch'] != 1
                or r['optimizer_steps'] != cfg['epochs'] or r['main_epochs'] + r['warmup_epochs'] != cfg['epochs']
                or r['structural_optimizer_steps'] != cfg['warmup']
                or r['integrated_optimizer_steps'] != cfg['epochs'] - cfg['warmup']):
            raise ValueError('Unequal or invalid optimizer-update budget')
        if cfg['eval_every'] != 1 or cfg['lr'] != .001 or cfg['num_negatives'] != 4:
            raise ValueError('Study requires every-epoch evaluation, lr=.001 and four negatives')
        if r['candidate_entities'] != cfg['max_entities']:
            raise ValueError('Candidate count differs from requested subset')
        history = r['history']
        if [row['epoch'] for row in history] != list(range(1, cfg['epochs'] + 1)):
            raise ValueError('Incomplete epoch history')
        for row in history:
            phase = 'structural_warmup' if row['epoch'] <= cfg['warmup'] else 'main'
            if row['phase'] != phase or not math.isfinite(row['loss']):
                raise ValueError('Invalid training phase/loss')
            if set(METRICS) - row['validation'].keys():
                raise ValueError('Missing validation/tie diagnostics')
        protocols.add(digest({k: v for k, v in cfg.items() if k not in ('variant', 'warmup', 'seed')}))
    if len(protocols) != 1 or len(enabled_warmups) != 1:
        raise ValueError('Experiment settings differ beyond architecture, warmup and seed')
    for seed in seeds:
        paired = [r for r in results if r['settings']['seed'] == seed]
        if len({r['initial_structural_sha256'] for r in paired}) != 1:
            raise ValueError('Structural initialization is not paired')
        for architecture in ('residual', 'residual-no-softprompt'):
            if len({r['initial_trainable_sha256'] for r in paired if r['settings']['variant'] == architecture}) != 1:
                raise ValueError('Warmup pair did not start with identical complete trainable parameters')


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


def analyze_run(r):
    history = r['history']
    main = [row for row in history if row['phase'] == 'main']
    warm = [row for row in history if row['phase'] == 'structural_warmup']
    tail = main[-min(5, len(main)):]
    loss_falls_rank_stalls = (len(tail) > 1 and tail[-1]['loss'] < tail[0]['loss']
                             and max(row['validation']['MRR'] for row in tail[1:]) <= tail[0]['validation']['MRR'] + 1e-6)
    flags = []
    tie_rows = [('selected_' + split, r[split]) for split in ('validation', 'test')]
    tie_rows += [(f"epoch_{row['epoch']}_{row['phase']}", row['validation']) for row in history]
    for label, m in tie_rows:
        if m['Tie_query_fraction'] >= .5 or m['Max_tied_candidates'] >= max(10, .1 * r['candidate_entities']):
            flags.append(label)
    return {'configuration': r['configuration'], 'seed': r['settings']['seed'],
            'warmup_loss': loss_behavior(warm), 'integrated_loss': loss_behavior(main),
            'final_training_loss': history[-1]['loss'], 'best_epoch': r['best_epoch'],
            'best_validation_MRR': r['validation']['MRR'],
            'final_epoch_validation_MRR': history[-1]['validation']['MRR'],
            'loss_falls_while_ranking_stalls_last_up_to_5_main_epochs': loss_falls_rank_stalls,
            'severe_tie_flags': flags, 'epoch_seconds': [h['epoch_seconds'] for h in history],
            'mean_epoch_seconds': stats.mean(h['epoch_seconds'] for h in history),
            'duration_seconds': r['duration_seconds'], 'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes'],
            'optimizer_steps': r['optimizer_steps']}


def paired_comparisons(results, seeds):
    indexed = {(r['configuration'], r['settings']['seed']): r for r in results}
    pairs = [('warmup_effect_residual', 'residual-warmup', 'residual-no-warmup'),
             ('warmup_effect_no_softprompt', 'residual-no-softprompt-warmup', 'residual-no-softprompt-no-warmup'),
             ('softprompt_effect_with_warmup', 'residual-warmup', 'residual-no-softprompt-warmup'),
             ('softprompt_effect_without_warmup', 'residual-no-warmup', 'residual-no-softprompt-no-warmup')]
    rows, summaries = [], []
    for question, a, b in pairs:
        for metric in ('validation_MRR', 'test_MRR', 'validation_Hits@10', 'test_Hits@10', 'duration_seconds'):
            differences = []
            av, bv = [], []
            for seed in seeds:
                def value(identifier):
                    r = indexed[identifier, seed]
                    if metric == 'duration_seconds':
                        return r[metric]
                    split, name = metric.split('_', 1)
                    return r[split][name]
                va, vb = value(a), value(b)
                av.append(va)
                bv.append(vb)
                differences.append(va - vb)
                rows.append({'question': question, 'A': a, 'B': b, 'seed': seed, 'metric': metric,
                             'A_value': va, 'B_value': vb, 'A_minus_B': va - vb,
                             'percent_change_vs_B': 100 * (va - vb) / vb if vb else None})
            summaries.append({'question': question, 'A': a, 'B': b, 'metric': metric,
                              **mean_sd(differences), 'mean_A': stats.mean(av), 'mean_B': stats.mean(bv),
                              'percent_change_of_means_vs_B': 100 * (stats.mean(av) - stats.mean(bv)) / stats.mean(bv)
                                  if stats.mean(bv) else None})
    return rows, summaries


def ablation_report(directory, results, seeds=(0, 1, 2)):
    validate_runs(results, seeds)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries, table, individual, ties = {}, [], [], []
    for identifier in CONFIGURATIONS:
        runs = sorted((r for r in results if r['configuration'] == identifier), key=lambda r: r['settings']['seed'])
        summary = {split: {m: mean_sd([r[split][m] for r in runs]) for m in METRICS} for split in ('validation', 'test')}
        summary.update(duration_seconds=mean_sd([r['duration_seconds'] for r in runs]),
                       mean_epoch_seconds=mean_sd([stats.mean(h['epoch_seconds'] for h in r['history']) for r in runs]),
                       final_training_loss=mean_sd([r['history'][-1]['loss'] for r in runs]),
                       final_epoch_validation_MRR=mean_sd([r['history'][-1]['validation']['MRR'] for r in runs]),
                       best_epochs=[r['best_epoch'] for r in runs],
                       peak_gpu_memory_bytes_max=max((r['peak_gpu_memory_bytes'] or 0 for r in runs), default=0) or None)
        summaries[identifier] = summary
        table.append({'configuration': identifier, 'architecture': runs[0]['settings']['variant'],
                      'warmup_epochs': runs[0]['warmup_epochs'], 'integrated_epochs': runs[0]['main_epochs'],
                      **{f'{split}_{m}_{stat}': entry[stat] for split in ('validation', 'test')
                         for m, entry in summary[split].items() for stat in ('mean', 'std')},
                      'duration_seconds_mean': summary['duration_seconds']['mean'],
                      'duration_seconds_std': summary['duration_seconds']['std'],
                      'mean_epoch_seconds': summary['mean_epoch_seconds']['mean'],
                      'peak_gpu_memory_bytes_max': summary['peak_gpu_memory_bytes_max']})
        for r in runs:
            individual.append({'configuration': identifier, 'seed': r['settings']['seed'],
                               'best_epoch': r['best_epoch'], 'final_training_loss': r['history'][-1]['loss'],
                               'final_epoch_validation_MRR': r['history'][-1]['validation']['MRR'],
                               'duration_seconds': r['duration_seconds'], 'optimizer_steps': r['optimizer_steps'],
                               'structural_updates': r['structural_optimizer_steps'],
                               'integrated_updates': r['integrated_optimizer_steps'],
                               'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes'],
                               **{f'{split}_{m}': r[split][m] for split in ('validation', 'test') for m in METRICS}})
            for label, metrics in [(split, r[split]) for split in ('validation', 'test')]:
                ties.append({'configuration': identifier, 'seed': r['settings']['seed'], 'evaluation': label,
                             **metrics, 'optimistic_minus_corrected_MRR': metrics['Optimistic_MRR'] - metrics['MRR']})
            for h in r['history']:
                ties.append({'configuration': identifier, 'seed': r['settings']['seed'],
                             'evaluation': 'epoch_validation', 'epoch': h['epoch'], 'phase': h['phase'],
                             **h['validation'], 'optimistic_minus_corrected_MRR': h['validation']['Optimistic_MRR'] - h['validation']['MRR']})
    winner = max(CONFIGURATIONS, key=lambda c: summaries[c]['validation']['MRR']['mean'])
    paired, paired_summary = paired_comparisons(results, seeds)
    summary = {'configurations': summaries, 'selected_by_validation': winner,
               'selected_test_results': summaries[winner]['test'], 'selection_metric': 'mean validation MRR',
               'seeds': list(seeds), 'manifest_sha256': results[0]['manifest_sha256'],
               'candidate_entities': results[0]['candidate_entities'], 'paired_comparisons': paired_summary,
               'note': 'Descriptive paired differences only; three seeds do not establish statistical significance. '
                       'Test metrics never select the configuration. Ties between means use registry order.',
               'phase_note': 'Warmup curves evaluate the structural model and structural BCE; main curves evaluate '
                             'the integrated model and integrated BCE. Never average across phases at one epoch.',
               'loss_diagnostics': 'OLS slope within phase; spikes are positive loss changes above max(median(delta)+3*MAD(delta),1e-8). '
                                   '95% reduction is relative to the observed within-phase minimum, not proof of convergence.',
               'tie_flag_rule': 'query fraction >= 0.5 OR maximum tied candidates >= max(10, 10% of candidates)',
               'timing_note': 'Total includes model preparation, evaluations and checkpoint I/O across resumptions; '
                              'epoch time includes training and scheduled validation. Excludes offline/preflight tests. '
                              'Frozen pooled cache hits can reduce preparation time; compare epoch time too.'}
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'runs.json', results)
    write_json(directory / 'training_behavior.json', [analyze_run(r) for r in results])
    write_json(directory / 'paired_comparisons.json', paired_summary)
    write_csv(directory / 'comparison.csv', table)
    write_csv(directory / 'runs.csv', individual)
    write_csv(directory / 'paired_seed_differences.csv', paired)
    write_csv(directory / 'ties.csv', ties)
    plot_ablation(directory, results, summaries)
    return summary


def plot_ablation(directory, results, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(CONFIGURATIONS, plt.get_cmap('tab10').colors))
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=150)
        plt.close(fig)
    for name, metric, title in ((PLOTS[0], None, 'BCE loss'), (PLOTS[1], 'MRR', 'Validation MRR'),
                                 (PLOTS[2], 'Hits@10', 'Validation Hits@10')):
        fig, ax = plt.subplots(figsize=(12, 6))
        for identifier in CONFIGURATIONS:
            runs = [r for r in results if r['configuration'] == identifier]
            for phase, style in (('structural_warmup', '--'), ('main', '-')):
                rows = [[h for h in r['history'] if h['phase'] == phase] for r in runs]
                if not rows[0]:
                    continue
                epochs = [h['epoch'] for h in rows[0]]
                values = [[h['loss'] if metric is None else h['validation'][metric] for h in row] for row in rows]
                means = [stats.mean(v) for v in zip(*values)]
                deviations = [stats.stdev(v) if len(v) > 1 else 0 for v in zip(*values)]
                ax.plot(epochs, means, style, color=colors[identifier], label=identifier + ' / ' + phase)
                ax.fill_between(epochs, [m-s for m, s in zip(means, deviations)],
                                [m+s for m, s in zip(means, deviations)], color=colors[identifier], alpha=.1)
        warm = max(r['warmup_epochs'] for r in results)
        ax.axvspan(.5, warm + .5, alpha=.06, color='grey', label='Warmup window for enabled configurations only')
        ax.set(xlabel='Total epoch', ylabel=title, title=title + ': mean +/- sample SD; dashed = structural, solid = integrated')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        finish(fig, name)
    for name, metric in ((PLOTS[3], 'MRR'), (PLOTS[4], 'Hits@10'), (PLOTS[7], 'duration_seconds')):
        fig, ax = plt.subplots(figsize=(12, 6))
        values = [summaries[c][metric] if metric == 'duration_seconds' else summaries[c]['test'][metric] for c in CONFIGURATIONS]
        ax.bar(list(CONFIGURATIONS), [v['mean'] for v in values], yerr=[v['std'] or 0 for v in values],
               color=[colors[c] for c in CONFIGURATIONS], capsize=4)
        ax.tick_params(axis='x', rotation=16)
        ax.set(ylabel=metric, title=('Total execution seconds' if metric == 'duration_seconds' else 'Validation-selected test ' + metric)
               + ' +/- sample SD')
        finish(fig, name)
    for name, metric in ((PLOTS[5], None), (PLOTS[6], 'MRR')):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, identifier in zip(axes.flat, CONFIGURATIONS):
            for r in (r for r in results if r['configuration'] == identifier):
                color = plt.get_cmap('tab10')(r['settings']['seed'] % 10)
                for phase, style in (('structural_warmup', '--'), ('main', '-')):
                    rows = [h for h in r['history'] if h['phase'] == phase]
                    if rows:
                        ax.plot([h['epoch'] for h in rows], [h['loss'] if metric is None else h['validation'][metric] for h in rows],
                                style, color=color, label=f"seed {r['settings']['seed']} / {phase}")
                if r['warmup_epochs']:
                    ax.axvline(r['warmup_epochs'] + .5, color='grey', alpha=.25)
            ax.set(title=identifier, xlabel='Total epoch', ylabel='BCE loss' if metric is None else 'Validation MRR')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        finish(fig, name)
