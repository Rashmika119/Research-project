"""Validated single-seed comparison for the six shared-RGAT experiments."""
from dataclasses import asdict
import math
from pathlib import Path
import statistics

from training.experiment_io import digest, write_json
from training.reports import METRICS, write_csv
from training.shared_study import CONFIGURATIONS, PROTOCOL, make_config, expected_budget

PLOTS = ('training_loss.png', 'validation_mrr.png', 'validation_hits10.png',
         'selected_metrics.png', 'time_memory.png', 'gate_effects.png')


def validate_runs(results, seed=0):
    expected = {(c, seed) for c in CONFIGURATIONS}
    actual = [(r['configuration'], r['settings']['seed']) for r in results]
    if len(actual) != 6 or set(actual) != expected:
        raise ValueError('Reports require exactly the six configurations for the requested seed')
    for key in ('manifest_sha256', 'dataset_sha256', 'environment', 'numerical_policy',
                'lm_identity', 'text_cache_keys', 'initial_structural_sha256', 'subset_statistics'):
        if any(r.get(key) is None for r in results) or len({digest(r[key]) for r in results}) != 1:
            raise ValueError('Mixed/missing study provenance: ' + key)
    for gated in (True, False):
        paired = [r for r in results if CONFIGURATIONS[r['configuration']][1] == gated]
        if len({r['initial_trainable_sha256'] for r in paired}) != 1:
            raise ValueError('Arrangements must share initial weights within each gate setting')
    numerical = set()
    for r in results:
        c, cfg = r['configuration'], r['settings']
        wanted = make_config(c, seed, **{k: cfg[k] for k in ('raw_dir', 'text_dir', 'text_batch_size', 'eval_batch_size')})
        if cfg != asdict(wanted):
            raise ValueError('Result does not use the shared-study settings')
        numerical.add(digest({k: v for k, v in cfg.items() if k != 'variant'}))
        counts = r['subset_statistics']
        if (r['status'] != 'complete' or r['initialization'] != 'scratch' or r['baseline_checkpoint_used']
                or r['architecture'] != c or r['dataset_scope'] != 'subset' or r['candidate_entities'] != 1000
                or counts['entities'] != 1000 or counts['relations'] != 237
                or any(counts[k] < 1 for k in ('train_triples', 'valid_triples', 'test_triples'))
                or r['ranking_policy'] != 'average_exact_ties' or r['warmup_epochs'] != 5 or r['main_epochs'] != 25):
            raise ValueError('Invalid scope, initialization, ranking or phase budgets')
        budget = expected_budget(c, counts['train_triples'])
        if any(r[k] != v for k, v in budget.items()):
            raise ValueError('Invalid optimizer budget')
        history = r['history']
        if [h['epoch'] for h in history] != list(range(1, 31)):
            raise ValueError('Incomplete epoch history')
        steps = budget['steps_per_epoch']
        batch_counts = [32768] * (steps-1) + [counts['train_triples'] - 32768*(steps-1)]
        eligible = []
        for h in history:
            e = h['epoch']
            warm = e <= 5
            if h['phase'] != ('structural_warmup' if warm else 'main') or h['optimizer_steps'] != e*steps:
                raise ValueError('Invalid phase/progress')
            updates = h['step_losses']
            if ([s['optimizer_step'] for s in updates] != list(range((e-1)*steps+1, e*steps+1))
                    or [s['positive_count'] for s in updates] != batch_counts
                    or any(not math.isfinite(s['loss']) for s in updates)
                    or not math.isclose(h['loss'], statistics.mean(s['loss'] for s in updates), rel_tol=1e-10, abs_tol=1e-10)):
                raise ValueError('Invalid step history or epoch loss')
            if ('validation' in h) != (e % 5 == 0):
                raise ValueError('Validation must run every five epochs')
            if 'validation' in h:
                if h['validation_model'] != ('structural' if warm else 'integrated'):
                    raise ValueError('Validation phase mislabeled')
                if not warm:
                    eligible.append(h)
        best = max(eligible, key=lambda h: h['validation']['MRR'])
        if r['best_epoch'] != best['epoch'] or not math.isclose(r['validation']['MRR'], best['validation']['MRR'], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError('Selection must use scheduled integrated validation only')
        if r['final_epoch_validation'] != history[-1]['validation']:
            raise ValueError('Incorrect final epoch validation')
        for values in (r['validation'], r['test'], *[h['validation'] for h in history if 'validation' in h]):
            if set(METRICS) - values.keys() or any(not math.isfinite(values[m]) for m in METRICS):
                raise ValueError('Missing/nonfinite ranking metrics')
        gated = CONFIGURATIONS[c][1]
        if ('refinement_gate' in r['learned_gates']) != gated:
            raise ValueError('Refinement gate does not match configuration')
    if len(numerical) != 1:
        raise ValueError('Optimization settings differ between configurations')


def shared_report(directory, results, seed=0):
    validate_runs(results, seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indexed = {r['configuration']: r for r in results}
    rows, epochs, steps = [], [], []
    for c, (arrangement, gated) in CONFIGURATIONS.items():
        r = indexed[c]
        rows.append({'configuration': c, 'seed': seed, 'arrangement': arrangement,
                     'refinement_gated': gated, 'unique_rgat_layers': 2,
                     'warmup_layer_calls': 2 if arrangement == 'c' else 1,
                     'integrated_layer_calls': {'a': 3, 'b': 2, 'c': 4}[arrangement],
                     'best_epoch': r['best_epoch'], 'optimizer_steps': r['optimizer_steps'],
                     'duration_seconds': r['duration_seconds'], 'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes'],
                     'refinement_coefficient': r['learned_gates'].get('refinement_gate', 1.),
                     **{f'{split}_{m}': r[split][m] for split in ('validation', 'test') for m in METRICS}})
        for h in r['history']:
            epochs.append({'configuration': c, 'seed': seed,
                           **{k: v for k, v in h.items() if k not in ('step_losses', 'validation')},
                           **{'validation_'+k: v for k, v in h.get('validation', {}).items()}})
            steps.extend({'configuration': c, 'seed': seed, 'epoch': h['epoch'], 'phase': h['phase'], **s}
                         for s in h['step_losses'])
    effects = []
    for a in 'abc':
        gated, plain = (indexed[f'shared-{a}-{g}'] for g in ('gated', 'ungated'))
        effects.append({'arrangement': a, **{f'{split}_{m}_gated_minus_ungated': gated[split][m]-plain[split][m]
                                            for split in ('validation', 'test') for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')}})
    winner = max(CONFIGURATIONS, key=lambda c: indexed[c]['validation']['MRR'])
    summary = {'protocol': PROTOCOL.replace('seed0', f'seed{seed}'), 'seeds': [seed],
               'selected_by_validation': winner, 'selected_test_results': indexed[winner]['test'],
               'configurations': rows, 'gate_effects': effects,
               'manifest_sha256': results[0]['manifest_sha256'], 'dataset_sha256': results[0]['dataset_sha256'],
               'interpretation': 'Single-seed subset results; no across-seed SD or significance claims. '
               'Ungated means U + V, not removal of the skip connection. '
               'A/B/C have different layer-call counts and C warms up both layers. '
               'Subset rankings cannot be compared directly with full-dataset rankings.'}
    for name, data in (('comparison.csv', rows), ('epochs.csv', epochs), ('steps.csv', steps), ('gate_effects.csv', effects)):
        write_csv(directory/name, data)
    write_json(directory/'runs.json', results)
    write_json(directory/'summary.json', summary)
    analysis = ['# Shared RGAT study', '', f'Selected by validation MRR: **{winner}**.', '', summary['interpretation'], '']
    analysis += [f"- {r['configuration']}: best epoch {r['best_epoch']}, validation MRR {r['validation_MRR']:.6f}, test MRR {r['test_MRR']:.6f}." for r in rows]
    analysis += ['', 'Gate effect (gated minus ungated; descriptive):']
    analysis += [f"- {r['arrangement'].upper()}: validation MRR {r['validation_MRR_gated_minus_ungated']:+.6f}; test MRR {r['test_MRR_gated_minus_ungated']:+.6f}." for r in effects]
    (directory/'research_analysis.md').write_text('\n'.join(analysis)+'\n', encoding='utf-8')
    plot_results(directory, indexed, rows, effects)
    return summary


def plot_results(directory, indexed, rows, effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory/name, dpi=150)
        plt.close(fig)

    for name, metric in zip(PLOTS[:3], (None, 'MRR', 'Hits@10')):
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (c, r) in enumerate(indexed.items()):
            for phase, style in (('structural_warmup', '--'), ('main', '-')):
                h = [h for h in r['history'] if h['phase'] == phase and (metric is None or 'validation' in h)]
                ax.plot([h['epoch'] for h in h], [h['validation'][metric] if metric else h['loss'] for h in h],
                        style, marker='o' if metric else None, color=plt.get_cmap('tab10')(i), label=f'{c}: {phase}')
        ax.axvline(5, color='grey', alpha=.5)
        ax.set(xlabel='Epoch', ylabel=metric or 'BCE loss', title='Dashed: structural warmup; solid: integrated')
        ax.legend(fontsize=6); ax.grid(alpha=.2)
        finish(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ('MRR', 'Hits@10')):
        for offset, split in ((-.2, 'validation'), (.2, 'test')):
            ax.bar([i+offset for i in range(6)], [r[f'{split}_{metric}'] for r in rows], width=.4, label=split)
        ax.set_xticks(range(6), [r['configuration'].replace('shared-', '') for r in rows], rotation=30)
        ax.set_title(metric); ax.legend()
    finish(fig, 'selected_metrics.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, scale in ((axes[0], 'duration_seconds', 1), (axes[1], 'peak_gpu_memory_bytes', 2**30)):
        ax.bar(range(6), [(r[metric] or 0)/scale for r in rows])
        ax.set_xticks(range(6), [r['configuration'].replace('shared-', '') for r in rows], rotation=30)
        ax.set_title('Peak allocated GPU GiB (0 when unavailable)' if scale != 1 else 'Total seconds including setup/evaluation')
    finish(fig, 'time_memory.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list('ABC'), [r['validation_MRR_gated_minus_ungated'] for r in effects])
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(ylabel='Validation MRR: gated minus ungated', title='Single-seed descriptive gate effects')
    finish(fig, 'gate_effects.png')
