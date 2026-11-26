"""Single-baseline reports from completed FB15K histories; no evaluation/training."""
import math
from pathlib import Path
import statistics

from training.baseline_fb15k import COUNTS, validate_config, expected_budget
from training.research_pipeline import ExperimentConfig
from training.experiment_io import write_json
from training.reports import METRICS, write_csv

PLOTS = ('training_loss', 'optimizer_step_loss', 'validation_metrics', 'gpu_memory')


def validate_result(r):
    cfg = r['settings']
    config = validate_config(ExperimentConfig(**cfg))
    budget = expected_budget(config)
    steps_per_epoch = budget['steps_per_epoch']
    batch = config.train_batch_size or COUNTS[2]
    sizes = [batch] * (steps_per_epoch - 1) + [budget['last_batch_size']]
    micro = config.micro_batch_size or batch
    micros = [(n + micro - 1) // micro for n in sizes]
    if (r['status'] != 'complete' or r['configuration'] != 'baseline'
            or r['architecture'] != 'baseline' or r['initialization'] != 'scratch'
            or r['baseline_checkpoint_used'] or r['dataset_scope'] != 'full'
            or r['candidate_entities'] != COUNTS[0] or r['ranking_policy'] != 'average_exact_ties'
            or r['final_epoch'] != config.epochs or r['warmup_epochs'] != 0 or r['main_epochs'] != config.epochs):
        raise ValueError('Expected a completed scratch FB15K baseline')
    actual = tuple(r['subset_statistics'][k] for k in
                   ('entities', 'relations', 'train_triples', 'valid_triples', 'test_triples'))
    if actual != COUNTS:
        raise ValueError('Wrong dataset counts')
    if any(r[k] is not None for k in ('lm_identity', 'text_cache_keys', 'initial_projection_sha256',
                                     'initial_validation', 'initial_validation_epoch')) or r['learned_gates']:
        raise ValueError('Baseline must not use text, gates or integrated validation')
    for k, v in budget.items():
        if k != 'last_batch_size' and r[k] != v:
            raise ValueError('Wrong update budget: ' + k)
    history = r['history']
    if [h['epoch'] for h in history] != list(range(1, config.epochs + 1)):
        raise ValueError('Incomplete epoch history')
    for h in history:
        epoch = h['epoch']
        steps = h['step_losses']
        if (h['phase'] != 'main' or h['optimizer_steps'] != epoch*steps_per_epoch
                or [s['optimizer_step'] for s in steps] != list(range((epoch-1)*steps_per_epoch+1, epoch*steps_per_epoch+1))
                or [s['positive_count'] for s in steps] != sizes
                or [s.get('micro_batches') for s in steps] != micros
                or any(not math.isfinite(s['loss']) for s in steps)
                or not math.isclose(h['loss'], statistics.mean(s['loss'] for s in steps), rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(h['example_weighted_loss'], sum(s['loss']*s['positive_count'] for s in steps)/COUNTS[2], rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError('Incorrect epoch/step/loss budget')
        if ('validation' in h) != (epoch % config.eval_every == 0 or epoch == config.epochs):
            raise ValueError('Incorrect validation schedule')
        if 'validation' in h and h['validation_model'] != 'baseline':
            raise ValueError('Incorrect validation model')
    eligible = [h for h in history if 'validation' in h]
    for m in [r['validation'], r['test'], *[h['validation'] for h in eligible]]:
        if any(k not in m or not math.isfinite(m[k]) for k in METRICS):
            raise ValueError('Missing or nonfinite ranking metrics')
    best = max(eligible, key=lambda h: h['validation']['MRR'])
    if r['best_epoch'] != best['epoch'] or r['validation'] != best['validation']:
        raise ValueError('Expected earliest best scheduled validation checkpoint')
    if r['final_epoch_validation'] != history[-1]['validation']:
        raise ValueError('Final validation does not match final epoch')
    if not r['environment'].get('source_sha256') or not r.get('manifest_sha256') or not r.get('dataset_sha256'):
        raise ValueError('Missing source/dataset identity')
    if r['numerical_policy'].get('deterministic_algorithms') is not True:
        raise ValueError('Expected strict numerical policy')
    d = r['architecture_diagnostics']
    if d['unique_rgat_layers'] != 2 or d['integrated_layer_executions'] != 2:
        raise ValueError('Baseline architecture changed')


def baseline_report(directory, result):
    validate_result(result)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(**result['settings'])
    budget = expected_budget(cfg)
    batch = cfg.train_batch_size or COUNTS[2]
    micro = cfg.micro_batch_size or batch
    summary = {'protocol': 'saved_fb15k_baseline_settings', 'settings': result['settings'], 'model': 'RGAT + ComplEx', 'dataset': 'FB15K',
               'seed': 0, 'complex_dimension': cfg.model['dim']//2, 'real_width': cfg.model['dim'],
               **{k: result[k] for k in ('best_epoch', 'final_epoch', 'validation', 'test',
                   'duration_seconds', 'peak_gpu_memory_bytes', 'peak_gpu_reserved_bytes',
                   'architecture_diagnostics', 'manifest_sha256', 'dataset_sha256', 'environment')},
               'budget': budget,
               'batching': {'effective_batch_size': batch, 'micro_batch_size': micro,
                            'accumulation_steps': cfg.accumulation_steps,
                            'micro_batches_per_epoch': (budget['steps_per_epoch']-1)*((batch+micro-1)//micro) + (budget['last_batch_size']+micro-1)//micro},
               'interpretation': 'One seed; no across-seed SD or significance claim. '
               'Existing two-layer RGAT + ComplEx, not pure ComplEx paper reproduction. '
               'Selected using validation only; test evaluated at the selected checkpoint.'}
    write_json(root/'summary.json', summary)
    write_csv(root/'metrics.csv', [{'split': split, 'best_epoch': result['best_epoch'], **result[split]}
                                 for split in ('validation', 'test')])
    write_csv(root/'epochs.csv', [{**{k: v for k, v in h.items() if k not in ('validation', 'step_losses', 'learned_gates')},
                                  **{'val_'+k: v for k, v in h.get('validation', {}).items()}}
                                 for h in result['history']])
    write_csv(root/'steps.csv', [{'epoch': h['epoch'], **s} for h in result['history'] for s in h['step_losses']])
    (root/'research_analysis.md').write_text(
        '# FB15K RGAT + ComplEx baseline\n\n'+summary['interpretation']+'\n\n'
        f"Best epoch: {result['best_epoch']}; validation MRR: {result['validation']['MRR']:.6f}; "
        f"test MRR: {result['test']['MRR']:.6f}.\n\n"
        f"{cfg.epochs} epochs, {budget['steps_per_epoch']} updates/epoch, {budget['optimizer_steps']} updates; "
        f"width{cfg.model['dim']}, {cfg.num_negatives} negatives. "
        f"Effective batch{batch}, micro-batch{micro}, accumulation{cfg.accumulation_steps}. "
        f"No warmup, text, scheduler or early stopping. Adam lr{cfg.lr}. "
        'Dataset differs from the previous FB15k-237 comparison; scores cannot be directly compared.\n',
        encoding='utf-8')
    plot_results(root, result)
    return summary


def plot_results(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = result['history']
    validation = [h for h in history if 'validation' in h]
    steps = [s for h in history for s in h['step_losses']]
    specs = {
        'training_loss': ('Epoch', 'BCE loss', [h['epoch'] for h in history],
                          {'mean update loss': [h['loss'] for h in history],
                           'example-weighted loss': [h['example_weighted_loss'] for h in history]}),
        'optimizer_step_loss': ('Update', 'BCE loss', [s['optimizer_step'] for s in steps],
                                {'loss': [s['loss'] for s in steps]}),
        'validation_metrics': ('Epoch', 'Filtered metric', [h['epoch'] for h in validation],
                               {m: [h['validation'][m] for h in validation] for m in ('MRR','Hits@1','Hits@3','Hits@10')}),
        'gpu_memory': ('Epoch', 'Peak GPU GiB', [h['epoch'] for h in history],
                       {label: [h.get(field)/2**30 if h.get(field) is not None else float('nan') for h in history]
                        for label, field in [('allocated','peak_gpu_memory_bytes'),('reserved','peak_gpu_reserved_bytes')]}),
    }
    for name, (xlabel, ylabel, x, series) in specs.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, y in series.items():
            ax.plot(x, y, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title='FB15K baseline; seed 0')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(root/f'{name}.{extension}', dpi=200)
        plt.close(fig)
