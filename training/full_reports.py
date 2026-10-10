"""Full-study reports with sparse validation and explicit optimization phases."""
import math
from pathlib import Path
import statistics as stats

from training.ablation_reports import METRICS, mean_sd, loss_behavior
from training.experiment_io import digest, write_json
from training.reports import write_csv

PLOTS = ('mean_training_loss.png', 'mean_validation_mrr.png', 'mean_validation_hits10.png',
         'test_mrr.png', 'test_hits10.png', 'per_seed_loss.png', 'per_seed_validation_mrr.png',
         'training_time.png', 'gpu_memory.png', 'optimizer_steps_loss.png',
         'validation_mrr.png', 'validation_hits10.png', 'hits1_hits3.png')


def validate_runs(results):
    from dataclasses import asdict
    from training.full_study import CONFIGURATIONS, SEEDS, make_config, expected_budget, COUNTS
    actual = [(r['configuration'], r['settings']['seed']) for r in results]
    expected = {(c, s) for c in CONFIGURATIONS for s in SEEDS}
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Reports require exactly all nine unique configuration/seed results')
    for field in ('manifest_sha256', 'dataset_sha256', 'candidate_entities', 'ranking_policy',
                  'environment', 'numerical_policy'):
        if len({digest(r[field]) for r in results}) != 1:
            raise ValueError('Mixed study provenance: ' + field)
    text_runs = [r for r in results if r['architecture'] != 'baseline']
    for field in ('lm_identity', 'text_cache_keys'):
        if any(not r.get(field) for r in text_runs) or len({digest(r[field]) for r in text_runs}) != 1:
            raise ValueError('Text runs have missing/mismatched frozen LM or pooled cache identities')
    numerical_configs = set()
    for r in results:
        cfg = r['settings']
        identifier = r['configuration']
        wanted = make_config(identifier, cfg['seed'], **{k: cfg[k] for k in
                             ('raw_dir', 'text_dir', 'text_batch_size', 'eval_batch_size')})
        if cfg != asdict(wanted):
            raise ValueError('Result does not use the approved full-study configuration')
        numerical_configs.add(digest({k: v for k, v in cfg.items() if k not in ('variant', 'warmup', 'seed')}))
        if (r['status'] != 'complete' or r['initialization'] != 'scratch'
                or r['baseline_checkpoint_used'] or r['dataset_scope'] != 'full'
                or r['candidate_entities'] != 14541 or r['ranking_policy'] != 'average_exact_ties'
                or r['architecture'] != cfg['variant']):
            raise ValueError('Only completed scratch full-data runs with corrected ranking may be compared')
        if cfg['variant'] == 'baseline' and (r['lm_identity'] is not None or r['text_cache_keys'] is not None):
            raise ValueError('Baseline must not use a language model/text cache')
        actual_counts = tuple(r['subset_statistics'][k] for k in
                              ('entities', 'relations', 'train_triples', 'valid_triples', 'test_triples'))
        if actual_counts != COUNTS:
            raise ValueError('Wrong dataset split counts')
        if r['warmup_epochs'] != cfg['warmup'] or r['main_epochs'] != 300 - cfg['warmup']:
            raise ValueError('Wrong phase epoch budgets')
        for k, v in expected_budget(identifier).items():
            if r[k] != v:
                raise ValueError('Wrong update budget: ' + k)
        history = r['history']
        if [h['epoch'] for h in history] != list(range(1, 301)):
            raise ValueError('Incomplete epoch history')
        eligible = []
        for h in history:
            e = h['epoch']
            warm = e <= cfg['warmup']
            if h['phase'] != ('structural_warmup' if warm else 'main') or h['optimizer_steps'] != e * 9:
                raise ValueError('Invalid phase or cumulative optimizer steps')
            if not math.isfinite(h['loss']):
                raise ValueError('Nonfinite loss')
            steps = h['step_losses']
            if ([s['optimizer_step'] for s in steps] != list(range((e-1)*9+1, e*9+1))
                    or [s['positive_count'] for s in steps] != [32768]*8 + [9971]
                    or any(not math.isfinite(s['loss']) for s in steps)):
                raise ValueError('Incomplete optimizer-step history or dropped final batch')
            if not math.isclose(h['loss'], stats.mean(s['loss'] for s in steps), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Epoch loss disagrees with optimizer-step mean')
            if ('validation' in h) != (e % 5 == 0):
                raise ValueError('Validation must follow the five-epoch schedule')
            if 'validation' in h:
                expected_model = 'baseline' if cfg['variant'] == 'baseline' else 'structural' if warm else 'integrated'
                if h['validation_model'] != expected_model:
                    raise ValueError('Validation model/phase mislabeled')
                if not warm:
                    eligible.append(h)
        best = max(eligible, key=lambda h: h['validation']['MRR'])
        if r['best_epoch'] != best['epoch'] or not math.isclose(r['validation']['MRR'], best['validation']['MRR'], abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError('Checkpoint was not selected by scheduled main-model validation')
        if r['final_epoch_validation'] != history[-1]['validation']:
            raise ValueError('Final validation is not the final epoch')
        for metrics in [r['validation'], r['test'], *[h['validation'] for h in history if 'validation' in h]]:
            if set(METRICS) - metrics.keys() or any(not math.isfinite(metrics[k]) for k in METRICS):
                raise ValueError('Missing or nonfinite ranking/tie metrics')
    if len(numerical_configs) != 1:
        raise ValueError('Settings differ beyond configuration/seed/warmup')
    for seed in SEEDS:
        paired = [r for r in results if r['settings']['seed'] == seed]
        if len({r['initial_structural_sha256'] for r in paired}) != 1:
            raise ValueError('Initial structural parameters are not paired')
        pair = [r for r in paired if r['architecture'] != 'baseline']
        if len({r['initial_trainable_sha256'] for r in pair}) != 1:
            raise ValueError('No-Softprompt warmup pair has different initial trainable parameters')


def phase_name(r, h):
    return 'baseline' if r['architecture'] == 'baseline' else ('warmup' if h['phase'] == 'structural_warmup' else 'integrated')


def behavior(r):
    histories = r['history']
    main = [h for h in histories if h['phase'] == 'main']
    validation = [h for h in main if 'validation' in h]
    # Compare the same epochs for all models, independently of warmup eligibility.
    matched = [h for h in validation if h['epoch'] >= 35]
    deltas = [b['validation']['MRR'] - a['validation']['MRR'] for a, b in zip(matched, matched[1:])]
    tail = validation[-6:]
    return {'configuration': r['configuration'], 'seed': r['settings']['seed'],
            'warmup_loss': loss_behavior([h for h in histories if h['phase'] == 'structural_warmup']),
            'main_loss': loss_behavior(main), 'main_model': 'baseline' if r['architecture'] == 'baseline' else 'integrated',
            'best_epoch': r['best_epoch'], 'best_before_final': r['best_epoch'] < 300,
            'best_to_final_validation_mrr_drop': r['validation']['MRR'] - r['final_epoch_validation']['MRR'],
            'loss_falls_while_validation_stalls_last_6_evaluations':
                len(tail) > 1 and tail[-1]['loss'] < tail[0]['loss'] and
                max(h['validation']['MRR'] for h in tail[1:]) <= tail[0]['validation']['MRR'] + 1e-6,
            'matched_epochs_35_300_mrr_delta_sd': stats.stdev(deltas),
            'matched_epochs_35_300_ranking_declines': sum(x < 0 for x in deltas),
            'first_epoch_at_95pct_observed_best_main_mrr': next(h['epoch'] for h in validation
                if h['validation']['MRR'] >= .95 * r['validation']['MRR']),
            'severe_tie_evaluations': [h['epoch'] for h in histories if 'validation' in h and
                (h['validation']['Tie_query_fraction'] >= .5 or h['validation']['Max_tied_candidates'] >= 1454.1)]}


def full_report(directory, results):
    from training.full_study import CONFIGURATIONS, SEEDS, PROTOCOL
    validate_runs(results)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries, table, individual, ties, epochs, steps = {}, [], [], [], [], []
    for c in CONFIGURATIONS:
        runs = sorted((r for r in results if r['configuration'] == c), key=lambda r: r['settings']['seed'])
        summary = {split: {m: mean_sd([r[split][m] for r in runs]) for m in METRICS} for split in ('validation', 'test')}
        summary.update(duration_seconds=mean_sd([r['duration_seconds'] for r in runs]),
                       training_seconds=mean_sd([sum(h['train_seconds'] for h in r['history']) for r in runs]),
                       peak_gpu_memory_bytes=mean_sd([r['peak_gpu_memory_bytes'] for r in runs]),
                       final_validation_MRR=mean_sd([r['final_epoch_validation']['MRR'] for r in runs]),
                       best_epochs=[r['best_epoch'] for r in runs])
        summaries[c] = summary
        table.append({'configuration': c, **{f'{split}_{m}_{s}': values[s]
            for split in ('validation', 'test') for m, values in summary[split].items() for s in ('mean', 'std')},
            **{f'{m}_{s}': summary[m][s] for m in ('duration_seconds', 'training_seconds', 'peak_gpu_memory_bytes', 'final_validation_MRR') for s in ('mean', 'std')},
            'best_epochs': str(summary['best_epochs'])})
        for r in runs:
            base = {'configuration': c, 'seed': r['settings']['seed']}
            individual.append({**base, **{k: r[k] for k in ('best_epoch', 'duration_seconds', 'peak_gpu_memory_bytes',
                'optimizer_steps', 'structural_optimizer_steps', 'integrated_optimizer_steps', 'baseline_optimizer_steps', 'final_training_loss')},
                'final_validation_MRR': r['final_epoch_validation']['MRR'],
                **{f'{split}_{m}': r[split][m] for split in ('validation', 'test') for m in METRICS}})
            evaluations = [(split, None, r[split]) for split in ('validation', 'test')]
            if r.get('initial_validation'):
                evaluations.append(('initial_integrated_diagnostic', r['warmup_epochs'], r['initial_validation']))
            for h in r['history']:
                epochs.append({**base, **{k: v for k, v in h.items() if k not in ('step_losses', 'validation')},
                               'model_phase': phase_name(r, h), **{'val_'+k: v for k, v in h.get('validation', {}).items()}})
                steps.extend({**base, 'epoch': h['epoch'], 'phase': phase_name(r, h), **s} for s in h['step_losses'])
                if 'validation' in h:
                    evaluations.append((phase_name(r, h), h['epoch'], h['validation']))
            for label, epoch, m in evaluations:
                ties.append({**base, 'evaluation': label, 'epoch': epoch, **m,
                             'optimistic_minus_corrected_MRR': m['Optimistic_MRR'] - m['MRR'],
                             'severe_ties': m['Tie_query_fraction'] >= .5 or m['Max_tied_candidates'] >= 1454.1})
    indexed = {(r['configuration'], r['settings']['seed']): r for r in results}
    paired, paired_summary = [], []
    contrasts = [('warmup_effect', 'residual-no-softprompt-warmup', 'residual-no-softprompt-no-warmup'),
                 ('text_with_warmup_vs_baseline', 'residual-no-softprompt-warmup', 'baseline'),
                 ('text_without_warmup_vs_baseline', 'residual-no-softprompt-no-warmup', 'baseline')]
    behaviors = [behavior(r) for r in results]
    behavior_index = {(r['configuration'], r['seed']): r for r in behaviors}
    for question, a, b in contrasts:
        names = [f'{split}_{m}' for split in ('validation', 'test') for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')]
        names += ['duration_seconds', 'matched_epochs_35_300_mrr_delta_sd', 'first_epoch_at_95pct_observed_best_main_mrr']
        for metric in names:
            av, bv = [], []
            for seed in SEEDS:
                def get(c):
                    if metric in behavior_index[c, seed]:
                        return behavior_index[c, seed][metric]
                    r = indexed[c, seed]
                    if metric in r:
                        return r[metric]
                    split, name = metric.split('_', 1)
                    return r[split][name]
                x, y = get(a), get(b)
                av.append(x); bv.append(y)
                paired.append({'question': question, 'A': a, 'B': b, 'seed': seed, 'metric': metric,
                               'A_value': x, 'B_value': y, 'difference_A_minus_B': x-y,
                               'percent_change_vs_B': 100*(x-y)/y if y else None})
            differences = [x-y for x, y in zip(av, bv)]
            paired_summary.append({'question': question, 'A': a, 'B': b, 'metric': metric,
                **mean_sd(differences), 'mean_A': stats.mean(av), 'mean_B': stats.mean(bv),
                'positive_differences': sum(v > 0 for v in differences), 'negative_differences': sum(v < 0 for v in differences),
                'percent_change_of_means_vs_B': 100*(stats.mean(av)-stats.mean(bv))/stats.mean(bv) if stats.mean(bv) else None})
    winner = max(CONFIGURATIONS, key=lambda c: summaries[c]['validation']['MRR']['mean'])
    summary = {'protocol': PROTOCOL, 'configurations': summaries, 'seeds': list(SEEDS),
               'selected_by_validation': winner, 'selection_metric': 'mean validation MRR',
               'selected_test_results': summaries[winner]['test'], 'paired_comparisons': paired_summary,
               'dataset_sha256': results[0]['dataset_sha256'], 'manifest_sha256': results[0]['manifest_sha256'],
               'candidate_entities': 14541,
               'interpretation': 'Descriptive paired effects only; three seeds do not establish significance. '
                   'Higher ranking metrics are better; lower duration/instability/convergence epoch may be better. '
                   'Convergence thresholds use each run\'s observed best and are not equal absolute targets. '
                   'Loss falling while validation worsens is an overfitting warning, not proof. '
                   'Ties between configuration means use configured order; checkpoints use earliest equal best.',
               'loss_definition': 'Epoch loss is the unweighted mean of nine update BCE losses. '
                   'Example-weighted loss is separately recorded; final batch has 9971 positives. '
                   'Warmup, integrated and baseline are labeled separately. Never pool phases.',
               'timing_definition': 'Total duration includes model/cache preparation, validation/test and checkpoint I/O; '
                   'training_seconds sums training epoch intervals. Cache-hit setup is cheaper. '
                   'Interrupted uncommitted work is not included in committed duration or step counts.',
               'validation_policy': 'Every five epochs; structural and initial integrated diagnostics cannot select. '
                   'Baseline/no-warmup selection starts at 5; warmup integrated selection starts at 35.'}
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'runs.json', results)
    write_json(directory / 'training_behavior.json', behaviors)
    write_json(directory / 'paired_comparisons.json', paired_summary)
    write_analysis(directory, summary, behaviors, ties)
    for name, rows in [('comparison.csv', table), ('runs.csv', individual), ('ties.csv', ties),
                       ('epochs.csv', epochs), ('steps.csv', steps), ('paired_seed_differences.csv', paired)]:
        write_csv(directory / name, rows)
    plot_results(directory, results, summaries)
    return summary


def write_analysis(directory, summary, behaviors, ties):
    """Descriptive answers derived only from completed, validated run records."""
    lines = ['# Research analysis', '',
             'Selection uses mean validation MRR. These are descriptive comparisons across three seeds, '
             'not evidence of statistical significance.', '',
             f"Selected configuration: **{summary['selected_by_validation']}**.", '',
             '## Does independent text improve ranking, and does warmup help?', '']
    for p in summary['paired_comparisons']:
        if p['metric'] == 'validation_MRR':
            direction = 'higher' if p['mean'] > 0 else 'lower' if p['mean'] < 0 else 'equal'
            lines.append(f"- {p['A']} versus {p['B']}: mean validation MRR is {direction}; "
                         f"paired delta {p['mean']:+.6f}, sample SD {p['std']:.6f}. "
                         f"Positive in {p['positive_differences']}/3 seeds and negative in "
                         f"{p['negative_differences']}/3. Improvement is "
                         + ('consistent in all three seeds.' if p['positive_differences'] == 3 else
                            'not positive in all three seeds.'))
    lines += ['', '## Convergence and ranking instability', '',
              'Convergence compares the epoch reaching 95% of each run\'s own observed best; '
              'it does not compare equal absolute ranking targets. Instability is the sample SD '
              'of validation-MRR changes at matched epochs 35–300. Lower values indicate earlier '
              'threshold attainment or less fluctuation, respectively.', '']
    for p in summary['paired_comparisons']:
        if p['question'] == 'warmup_effect' and p['metric'] in (
                'first_epoch_at_95pct_observed_best_main_mrr', 'matched_epochs_35_300_mrr_delta_sd'):
            lines.append(f"- {p['metric']}: warmup minus no-warmup {p['mean']:+.6f} "
                         f"(sample SD {p['std']:.6f}); negative in {p['negative_differences']}/3 seeds.")
    lines += ['', '## Does validation peak early; is there evidence suggesting overfitting?', '',
              'Falling training loss with stalled/worsening validation is a warning, not proof of overfitting.', '']
    for b in behaviors:
        lines.append(f"- {b['configuration']}, seed {b['seed']}: best epoch {b['best_epoch']}; "
                     f"best-to-final validation MRR drop {b['best_to_final_validation_mrr_drop']:.6f}; "
                     f"late falling-loss/stalled-validation flag: "
                     f"{b['loss_falls_while_validation_stalls_last_6_evaluations']}.")
    lines += ['', '## Computational cost', '',
              'Total durations include setup, cache, validation/test and checkpoint I/O; '
              'cache hits make later preparation cheaper. Epoch training intervals are also reported.', '']
    for c, s in summary['configurations'].items():
        lines.append(f"- {c}: mean total {s['duration_seconds']['mean']:.1f}s "
                     f"(sample SD {s['duration_seconds']['std']:.1f}s), "
                     f"mean training intervals {s['training_seconds']['mean']:.1f}s, "
                     f"mean peak allocated GPU {s['peak_gpu_memory_bytes']['mean']/2**30:.2f} GiB.")
    lines += ['', '## Are scores affected by ties?', '',
              'Corrected metrics use average exact-tie ranks. Optimistic-minus-corrected MRR measures '
              'the effect of using optimistic tie handling; severe-tie flags are descriptive.', '']
    for t in ties:
        if t['evaluation'] in ('validation', 'test'):
            lines.append(f"- {t['configuration']}, seed {t['seed']}, {t['evaluation']}: tied-query fraction "
                         f"{t['Tie_query_fraction']:.6f}, mean/max tied candidates "
                         f"{t['Mean_tied_candidates']:.3f}/{t['Max_tied_candidates']}, optimistic MRR gap "
                         f"{t['optimistic_minus_corrected_MRR']:.6f}; severe flag {t['severe_ties']}.")
    lines += ['', '## Test results after validation selection', '']
    for c, s in summary['configurations'].items():
        lines.append(f"- {c}: test MRR {s['test']['MRR']['mean']:.6f} +/- {s['test']['MRR']['std']:.6f}; "
                     f"test Hits@10 {s['test']['Hits@10']['mean']:.6f} +/- {s['test']['Hits@10']['std']:.6f} "
                     '(mean +/- sample SD).')
    (directory / 'research_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_results(directory, results, summaries):
    from training.full_study import CONFIGURATIONS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(CONFIGURATIONS, plt.get_cmap('tab10').colors))
    def finish(fig, name):
        fig.tight_layout(); fig.savefig(directory / name, dpi=150); plt.close(fig)
    styles = {'warmup': '--', 'integrated': '-', 'baseline': ':'}
    for name, metric, per_seed, use_steps in [
        ('mean_training_loss.png', None, False, False), ('mean_validation_mrr.png', 'MRR', False, False),
        ('mean_validation_hits10.png', 'Hits@10', False, False), ('per_seed_loss.png', None, True, False),
        ('per_seed_validation_mrr.png', 'MRR', True, False), ('optimizer_steps_loss.png', None, True, True)]:
        fig, axes = plt.subplots(3 if per_seed else 1, 1, figsize=(12, 12 if per_seed else 6), squeeze=False)
        for i, c in enumerate(CONFIGURATIONS):
            ax = axes[i if per_seed else 0, 0]
            runs = sorted((r for r in results if r['configuration'] == c), key=lambda r: r['settings']['seed'])
            for phase, style in styles.items():
                grouped = []
                for r in runs:
                    rows = [h for h in r['history'] if phase_name(r, h) == phase and (metric is None or 'validation' in h)]
                    if not rows:
                        continue
                    x = [s['optimizer_step'] for h in rows for s in h['step_losses']] if use_steps else [h['epoch'] for h in rows]
                    y = [s['loss'] for h in rows for s in h['step_losses']] if use_steps else [h['loss'] if metric is None else h['validation'][metric] for h in rows]
                    grouped.append(y)
                    if per_seed:
                        ax.plot(x, y, style, label=f"seed {r['settings']['seed']} / {phase}", color=plt.get_cmap('tab10')(r['settings']['seed']))
                if grouped and not per_seed:
                    means = [stats.mean(v) for v in zip(*grouped)]
                    sd = [stats.stdev(v) for v in zip(*grouped)]
                    ax.plot(x, means, style, label=c+' / '+phase, color=colors[c])
                    ax.fill_between(x, [m-s for m,s in zip(means, sd)], [m+s for m,s in zip(means, sd)], color=colors[c], alpha=.1)
            if not per_seed or c.endswith('-warmup') and not c.endswith('-no-warmup'):
                ax.axvline(270 if use_steps else 30, color='grey', alpha=.3)
            ax.set(xlabel='Optimizer updates' if use_steps else 'Epoch', ylabel=metric or 'BCE loss',
                   title=c if per_seed else 'Mean +/- sample SD; dashed warmup / solid integrated / dotted baseline')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        finish(fig, name)
    for name, split, metric in [('test_mrr.png', 'test', 'MRR'), ('test_hits10.png', 'test', 'Hits@10'),
        ('validation_mrr.png', 'validation', 'MRR'), ('validation_hits10.png', 'validation', 'Hits@10'),
        ('training_time.png', None, 'duration_seconds'), ('gpu_memory.png', None, 'peak_gpu_memory_bytes')]:
        fig, ax = plt.subplots(figsize=(12, 6))
        values = [summaries[c][split][metric] if split else summaries[c][metric] for c in CONFIGURATIONS]
        scale = 2**30 if metric == 'peak_gpu_memory_bytes' else 1
        ax.bar(list(CONFIGURATIONS), [v['mean']/scale for v in values], yerr=[v['std']/scale for v in values], capsize=4)
        ax.tick_params(axis='x', rotation=12)
        ax.set(ylabel='GiB' if scale != 1 else metric, title=f'{split or "Execution"} {metric}: mean +/- sample SD')
        finish(fig, name)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (split, metric) in zip(axes.flat, [(s,m) for s in ('validation','test') for m in ('Hits@1','Hits@3')]):
        v = [summaries[c][split][metric] for c in CONFIGURATIONS]
        ax.bar(range(3), [x['mean'] for x in v], yerr=[x['std'] for x in v], capsize=4)
        ax.set_xticks(range(3), ['Baseline', 'Text + warmup', 'Text no warmup']); ax.set_title(split+' '+metric)
    finish(fig, 'hits1_hits3.png')
