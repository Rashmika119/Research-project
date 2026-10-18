"""Full-dataset baseline-versus-Shared-C-gated reports from completed measured histories."""
import math
import statistics as stats
from pathlib import Path

from training.experiment_io import digest, write_json
from training.reports import METRICS, mean_sd, write_csv
from training.reports import behavior
from training.final_study import CONFIGURATIONS, PROTOCOL, SEEDS

CONTRASTS = (('shared_c_vs_baseline', 'shared-c-gated', 'baseline'),)
PLOTS = ('training_loss', 'optimizer_step_loss', 'validation_mrr_curve', 'validation_hits1_curve',
         'validation_hits3_curve', 'validation_hits10_curve', 'per_seed_validation',
         'validation_mrr', 'test_mrr', 'test_hits10', 'runtime', 'gpu_memory',
         'warmup_transition', 'refinement_gate', 'parameters')


def validate_runs(results, seeds=SEEDS):
    from dataclasses import asdict
    from training.final_study import CONFIGURATIONS, make_config, expected_budget, COUNTS, validate_seeds
    seeds = validate_seeds(seeds)
    actual = [(r['configuration'], r['settings']['seed']) for r in results]
    expected = {(c, s) for c in CONFIGURATIONS for s in seeds}
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Reports require exactly all requested configuration/seed results')
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
        if r['warmup_epochs'] != cfg['warmup'] or r['main_epochs'] != 100 - cfg['warmup']:
            raise ValueError('Wrong phase epoch budgets')
        for k, v in expected_budget(identifier).items():
            if r[k] != v:
                raise ValueError('Wrong update budget: ' + k)
        history = r['history']
        if [h['epoch'] for h in history] != list(range(1, 101)):
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
    for seed in seeds:
        paired = [r for r in results if r['settings']['seed'] == seed]
        if len({r['initial_structural_sha256'] for r in paired}) != 1:
            raise ValueError('Initial structural parameters are not paired')
    for r in results:
        baseline = r['architecture'] == 'baseline'
        if (baseline and r.get('initial_projection_sha256') is not None) or (not baseline and not r.get('initial_projection_sha256')):
            raise ValueError('Incorrect projection identity for architecture')
        expected_gates = set() if baseline else {'entity_text_gate', 'relation_text_gate', 'refinement_gate'}
        if set(r['learned_gates']) != expected_gates:
            raise ValueError('Incorrect text/refinement gates')
        d = r['architecture_diagnostics']
        if (d['unique_rgat_layers'] != 2
                or d['warmup_layer_executions'] != 2 or d['integrated_layer_executions'] != (2 if baseline else 4)
                or d['trainable_parameters'] < 1 or r['final_epoch'] != 100):
            raise ValueError('Invalid model inventory or final epoch')
        if (not r.get('gpu_memory_start') or r['gpu_memory_start']['free_bytes'] <= 0
                or r['gpu_memory_start']['total_bytes'] < r['gpu_memory_start']['free_bytes']
                or not r['peak_gpu_reserved_bytes'] or not r['peak_gpu_memory_bytes']
                or r['peak_gpu_reserved_bytes'] < r['peak_gpu_memory_bytes']):
            raise ValueError('Final GPU runs require available/allocated/reserved memory measurements')
        if not r['environment'].get('git_commit') or not r['environment'].get('source_sha256'):
            raise ValueError('Missing source identity')
        if r['numerical_policy'].get('deterministic_algorithms') is not True:
            raise ValueError('Final study requires strict determinism')
        if baseline:
            if r['initial_validation_epoch'] is not None or r['initial_validation'] is not None:
                raise ValueError('Baseline has no integrated-boundary validation')
        elif r['initial_validation_epoch'] != 10 or not r['initial_validation']:
            raise ValueError('Missing diagnostic warmup-boundary validation')
        for h in r['history']:
            weighted = sum(s['loss']*s['positive_count'] for s in h['step_losses']) / 272115
            if (not math.isclose(h['example_weighted_loss'], weighted, rel_tol=1e-10, abs_tol=1e-10)
                    or set(h['learned_gates']) != set(r['learned_gates'])
                    or any(not math.isfinite(x) for x in h['learned_gates'].values())
                    or not math.isfinite(h['epoch_seconds']) or h['epoch_seconds'] < 0
                    or not h['peak_gpu_reserved_bytes']):
                raise ValueError('Incomplete epoch diagnostics or incorrect weighted loss')
    for c in CONFIGURATIONS:
        if len({r['architecture_diagnostics']['trainable_parameters'] for r in results if r['configuration'] == c}) != 1:
            raise ValueError('Parameter count changes across seeds')
    for seed in seeds:
        p = {r['configuration']: r['architecture_diagnostics']['trainable_parameters'] for r in results if r['settings']['seed'] == seed}
        if not (p['shared-c-gated'] > p['baseline']):
            raise ValueError('Unexpected baseline/shared parameter counts')



def final_report(directory, results, seeds=SEEDS):
    validate_runs(results, seeds)
    seeds = tuple(seeds)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indexed = {(r['configuration'], r['settings']['seed']): r for r in results}
    behaviors = [behavior(r) for r in results]
    by_behavior = {(b['configuration'], b['seed']): b for b in behaviors}
    summaries, runs, epochs, steps, evaluations = {}, [], [], [], []
    for c in CONFIGURATIONS:
        group = [indexed[c, s] for s in seeds]
        summaries[c] = {split: {m: mean_sd([r[split][m] for r in group]) for m in METRICS}
                        for split in ('validation', 'test')}
        for field in ('duration_seconds', 'peak_gpu_memory_bytes', 'peak_gpu_reserved_bytes', 'best_epoch'):
            summaries[c][field] = mean_sd([r[field] for r in group])
        for field in ('trainable_parameters', 'unique_rgat_layers', 'integrated_layer_executions', 'warmup_layer_executions'):
            summaries[c][field] = group[0]['architecture_diagnostics'][field]
        summaries[c]['training_seconds'] = mean_sd([sum(h['train_seconds'] for h in r['history']) for r in group])
        for r in group:
            base = {'configuration': c, 'seed': r['settings']['seed']}
            runs.append({**base, **r['architecture_diagnostics'], **{k: r[k] for k in (
                'best_epoch', 'final_epoch', 'optimizer_steps', 'structural_optimizer_steps',
                'integrated_optimizer_steps', 'baseline_optimizer_steps', 'duration_seconds', 'peak_gpu_memory_bytes', 'peak_gpu_reserved_bytes')},
                **{'gpu_start_'+k: v for k, v in r['gpu_memory_start'].items()},
                'training_seconds': sum(h['train_seconds'] for h in r['history']),
                'initial_structural_sha256': r['initial_structural_sha256'],
                'initial_projection_sha256': r['initial_projection_sha256'],
                **{f'{split}_{m}': r[split][m] for split in ('validation', 'test') for m in METRICS}})
            for h in r['history']:
                epochs.append({**base, **{k: v for k, v in h.items() if k not in ('validation', 'step_losses', 'learned_gates')},
                               **{'val_'+k: v for k, v in h.get('validation', {}).items()}, **h['learned_gates']})
                steps.extend({**base, 'epoch': h['epoch'], 'phase': h['phase'], **s} for s in h['step_losses'])
                if 'validation' in h:
                    evaluations.append({**base, 'split': 'validation', 'epoch': h['epoch'], 'phase': h['phase'], **h['validation']})
            evaluations.extend({**base, 'split': split, 'epoch': r['best_epoch'], 'phase': 'selected', **r[split]}
                               for split in ('validation', 'test'))
            if r['initial_validation'] is not None:
                evaluations.append({**base, 'split': 'validation', 'epoch': 10, 'phase': 'initial_integrated_diagnostic', **r['initial_validation']})
    paired, contrasts = [], []
    for name, a, b in CONTRASTS:
        metrics = [f'{split}_{m}' for split in ('validation', 'test') for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')]
        metrics += ['duration_seconds', 'peak_gpu_memory_bytes', 'peak_gpu_reserved_bytes', 'trainable_parameters',
                    'training_seconds', 'first_epoch_at_95pct_observed_best_main_mrr']
        for metric in metrics:
            differences = []
            for seed in seeds:
                def get(c):
                    r = indexed[c, seed]
                    if metric in by_behavior[c, seed]: return by_behavior[c, seed][metric]
                    if metric in r['architecture_diagnostics']: return r['architecture_diagnostics'][metric]
                    if metric == 'training_seconds': return sum(h['train_seconds'] for h in r['history'])
                    if metric in r: return r[metric]
                    split, m = metric.split('_', 1)
                    return r[split][m]
                x, y = get(a), get(b)
                differences.append(x-y)
                paired.append({'contrast': name, 'A': a, 'B': b, 'seed': seed, 'metric': metric,
                               'A_value': x, 'B_value': y, 'difference_A_minus_B': x-y})
            contrasts.append({'contrast': name, 'A': a, 'B': b, 'metric': metric, **mean_sd(differences),
                              'positive_seeds': sum(x > 0 for x in differences), 'negative_seeds': sum(x < 0 for x in differences)})
    winner = max(CONFIGURATIONS, key=lambda c: summaries[c]['validation']['MRR']['mean'])
    seed_winners = {str(s): max(CONFIGURATIONS, key=lambda c: indexed[c, s]['validation']['MRR']) for s in seeds}
    summary = {'protocol': PROTOCOL, 'seeds': list(seeds), 'configurations': summaries,
               'selected_by_validation': winner, 'selection_metric': 'mean validation MRR',
               'selected_test_results': summaries[winner]['test'], 'per_seed_validation_winners': seed_winners,
               'paired_comparisons': contrasts, 'candidate_entities': 14541,
               'manifest_sha256': results[0]['manifest_sha256'], 'dataset_sha256': results[0]['dataset_sha256'],
               'interpretation': 'Seed 0 only; across-seed SD is unavailable, with no significance or seed-consistency claims. '
               'Warmup and integrated curves are separated. Test never selects epochs or architecture. '
               'Baseline trains structurally for 100 epochs; Shared C gated uses 10 warmup plus 90 integrated epochs. '
               'This compares complete models, not an isolated effect of sharing, text or warmup. Equal updates do not mean equal compute.'}
    table = []
    for c, s in summaries.items():
        row = {'configuration': c}
        for split in ('validation', 'test'):
            row.update({f'{split}_{m}_{k}': v for m, values in s[split].items() for k, v in values.items()})
        for field, values in s.items():
            if field in ('validation', 'test'): continue
            if isinstance(values, dict): row.update({f'{field}_{k}': v for k, v in values.items()})
            else: row[field] = values
        table.append(row)
    for name, rows in [('comparison.csv', table), ('runs.csv', runs), ('epochs.csv', epochs), ('steps.csv', steps),
                       ('evaluations.csv', evaluations), ('paired_seed_differences.csv', paired), ('paired_summary.csv', contrasts)]:
        write_csv(directory/name, rows)
    for split in ('validation', 'test'):
        write_csv(directory/f'per_seed_{split}.csv', [{'configuration': r['configuration'], 'seed': r['settings']['seed'],
                                                     'best_epoch': r['best_epoch'], **r[split]} for r in results])
    write_json(directory/'summary.json', summary)
    write_json(directory/'runs.json', results)
    write_json(directory/'training_behavior.json', behaviors)
    write_analysis(directory, summary, behaviors)
    plot_results(directory, indexed, summary)
    return summary


def write_analysis(directory, summary, behaviors):
    lines = ['# Baseline versus Shared C gated', '', summary['interpretation'], '',
             f"Selected by validation MRR: **{summary['selected_by_validation']}**.", '',
             '## Measured differences (Shared C gated minus baseline)', '']
    for p in summary['paired_comparisons']:
        lines.append(f"- {p['metric']}: {p['mean']:+.6f}; across-seed SD unavailable.")
    lines += ['', '## Parameters and cost', '']
    for c, v in summary['configurations'].items():
        lines.append(f"- {c}: {v['trainable_parameters']:,} trainable parameters, "
                     f"{v['unique_rgat_layers']} unique RGAT layers, "
                     f"{v['integrated_layer_executions']} layer executions per main-phase encoding; "
                     f"training seconds {v['training_seconds']['mean']:.3f}.")
    lines += ['', 'Total time includes preparation, cache I/O, checkpoints and evaluation. '
              'The baseline needs no language model or text cache. Shared C reuses two layers, '
              'but executes four graph layers per integrated encoding. GPU peaks include setup; '
              'weight sharing does not halve activation memory.', '', '## Learning behavior', '']
    for b in behaviors:
        lines.append(f"- {b['configuration']}: best epoch {b['best_epoch']}; best-to-final validation MRR "
                     f"drop {b['best_to_final_validation_mrr_drop']:.6f}; first epoch at 95% of its own "
                     f"observed best MRR: {b['first_epoch_at_95pct_observed_best_main_mrr']}.")
    lines += ['', 'The 95% threshold is relative to each model, not a common quality target. '
              'Structural weights are paired at scratch initialization. Shared C additionally learns '
              'text projection and three gates, and reuses its warmup-trained layers for refinement. '
              'A difference cannot be attributed exclusively to any one of those changes. '
              'Baseline checkpoint selection begins at epoch 5; Shared C selection begins at epoch 15. '
              'Test scores do not select the checkpoint or model. Seed 0 alone cannot establish significance.']
    (directory/'research_analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot_results(directory, indexed, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds = summary['seeds']
    colors = dict(zip(CONFIGURATIONS, plt.get_cmap('tab10').colors))
    labels = dict(zip(CONFIGURATIONS, ('KG-only baseline', 'Shared C gated')))
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory/(name+'.png'), dpi=300)
        fig.savefig(directory/(name+'.pdf'))
        plt.close(fig)
    specs = [('training_loss', None, False), ('optimizer_step_loss', None, True)]
    specs += [(f'validation_{name}_curve', m, False) for name, m in [('mrr','MRR'),('hits1','Hits@1'),('hits3','Hits@3'),('hits10','Hits@10')]]
    for name, metric, updates in specs:
        fig, ax = plt.subplots(figsize=(10,5))
        for c in CONFIGURATIONS:
            for phase, style in (('structural_warmup','--'),('main','-')):
                groups=[]
                for seed in seeds:
                    rows=[h for h in indexed[c,seed]['history'] if h['phase']==phase and (not metric or 'validation' in h)]
                    x=[s['optimizer_step'] for h in rows for s in h['step_losses']] if updates else [h['epoch'] for h in rows]
                    groups.append([s['loss'] for h in rows for s in h['step_losses']] if updates else [h['validation'][metric] if metric else h['loss'] for h in rows])
                if not x: continue
                values=[mean_sd(v) for v in zip(*groups)]
                ax.plot(x,[v['mean'] for v in values],style,color=colors[c],label=labels[c]+' / '+phase)
                if len(seeds)>1:
                    ax.fill_between(x,[v['mean']-v['std'] for v in values],[v['mean']+v['std'] for v in values],color=colors[c],alpha=.12)
        ax.axvline(90 if updates else 10,color='grey',alpha=.4)
        ax.set(xlabel='Optimizer update' if updates else 'Epoch',ylabel=metric or 'BCE loss',title='Seed 0; dashed: Shared C warmup; solid: main phase')
        ax.legend(fontsize=7); ax.grid(alpha=.2); finish(fig,name)
    fig, axes=plt.subplots(len(seeds),1,figsize=(10,4*len(seeds)),squeeze=False)
    for seed,ax in zip(seeds,axes[:,0]):
        for c in CONFIGURATIONS:
            rows=[h for h in indexed[c,seed]['history'] if h['phase']=='main' and 'validation' in h]
            ax.plot([h['epoch'] for h in rows],[h['validation']['MRR'] for h in rows],label=labels[c],color=colors[c])
        ax.set(title=f'Seed {seed}: main-phase validation',xlabel='Epoch',ylabel='MRR');ax.legend()
    finish(fig,'per_seed_validation')
    for name,split,metric in [('validation_mrr','validation','MRR'),('test_mrr','test','MRR'),('test_hits10','test','Hits@10'),('runtime',None,'training_seconds')]:
        fig,ax=plt.subplots(figsize=(9,5))
        vals=[summary['configurations'][c][split][metric] if split else summary['configurations'][c][metric] for c in CONFIGURATIONS]
        ax.bar(list(labels.values()),[v['mean'] for v in vals],yerr=[v['std'] for v in vals] if len(seeds)>1 else None,capsize=4)
        ax.set(ylabel=metric,title='Seed 0; across-seed SD unavailable');finish(fig,name)
    fig,ax=plt.subplots(figsize=(9,5))
    for offset,field,label in [(-.2,'peak_gpu_memory_bytes','allocated'),(.2,'peak_gpu_reserved_bytes','reserved')]:
        vals=[summary['configurations'][c][field] for c in CONFIGURATIONS]
        ax.bar([i+offset for i in range(len(CONFIGURATIONS))],[v['mean']/2**30 for v in vals],width=.4,yerr=[v['std']/2**30 for v in vals] if len(seeds)>1 else None,label=label,capsize=3)
    ax.set_xticks(range(len(CONFIGURATIONS)),list(labels.values()));ax.set_ylabel('Peak GPU GiB');ax.legend();finish(fig,'gpu_memory')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(list(labels.values()),[summary['configurations'][c]['trainable_parameters'] for c in CONFIGURATIONS])
    ax.set_ylabel('Trainable parameters');finish(fig,'parameters')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for c in CONFIGURATIONS:
        for seed in seeds:
            r=indexed[c,seed]
            if r['architecture'] == 'baseline': continue
            structural=[h for h in r['history'] if h['epoch']==10][0]['validation']['MRR']
            integrated=[h for h in r['history'] if h['epoch']==15][0]['validation']['MRR']
            axes[0].scatter([0],[structural],color=colors[c],alpha=.6)
            axes[1].plot([10,15],[r['initial_validation']['MRR'],integrated],color=colors[c],alpha=.6,label=labels[c] if seed==seeds[0] else None)
    axes[0].set(title='Warmup epoch 10 (structural only)',ylabel='Validation MRR',xticks=[])
    axes[1].set(title='Integrated boundary diagnostic to epoch 15',xlabel='Epoch',ylabel='Validation MRR');axes[1].legend(fontsize=7)
    finish(fig,'warmup_transition')
    fig,ax=plt.subplots(figsize=(10,5))
    for c in CONFIGURATIONS:
        for seed in seeds:
            if c == 'baseline': continue
            rows=indexed[c,seed]['history']
            ax.plot([h['epoch'] for h in rows],[h['learned_gates']['refinement_gate'] for h in rows],color=colors[c],alpha=.65,label=labels[c] if seed==seeds[0] else None)
    ax.axvline(10,color='grey',alpha=.4);ax.set(xlabel='Epoch',ylabel='Refinement coefficient: tanh(gF)');ax.legend()
    finish(fig,'refinement_gate')
