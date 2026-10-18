"""Fabricated report records used only in temporary test directories."""
from dataclasses import asdict
from training.final_study import CONFIGURATIONS,SEEDS,COUNTS,make_config,expected_budget

def fixture_results():
    def metrics(mrr):
        return {'MRR': mrr, 'Hits@1': mrr/2, 'Hits@3': mrr, 'Hits@10': mrr*2,
                'Optimistic_MRR': mrr+.001, 'Tie_query_fraction': .01,
                'Mean_tied_candidates': 1.01, 'Max_tied_candidates': 2}
    results = []
    for index, c in enumerate(CONFIGURATIONS):
        for seed in SEEDS:
            cfg = make_config(c, seed)
            history = []
            for e in range(1, 101):
                h = {'epoch': e, 'phase': 'structural_warmup' if e <= cfg.warmup else 'main',
                     'loss': 1.5-e*.001, 'example_weighted_loss': 1.5-e*.001,
                     'optimizer_steps': e*9, 'train_seconds': 1., 'epoch_seconds': 2.,
                     'step_losses': [{'optimizer_step': (e-1)*9+s+1, 'loss': 1.5-e*.001,
                                      'positive_count': 32768 if s < 8 else 9971} for s in range(9)]}
                if e % 5 == 0:
                    h['validation'] = metrics(.8 if e <= cfg.warmup else
                        .05 + index*.01 + seed*.001 + (min(e,75)-max(0,e-75)*.1)*.0001)
                    h['validation_model'] = 'baseline' if c == 'baseline' else 'structural' if e <= cfg.warmup else 'integrated'
                history.append(h)
            results.append({'configuration': c, 'architecture': cfg.variant, 'settings': asdict(cfg),
                'status': 'complete', 'initialization': 'scratch', 'baseline_checkpoint_used': False,
                'dataset_scope': 'full', 'candidate_entities': 14541, 'ranking_policy': 'average_exact_ties',
                'manifest_sha256': 'fabricated-manifest', 'dataset_sha256': 'fabricated-data',
                'environment': {'source_sha256': 'fixture'}, 'numerical_policy': {'fixture': True},
                'lm_identity': None if c == 'baseline' else {'name': 'fixture-LM'},
                'text_cache_keys': None if c == 'baseline' else {'entities': 'fixture', 'relations': 'fixture'},
                'initial_structural_sha256': f'paired-structural-{seed}',
                'initial_trainable_sha256': f'{cfg.variant}-{seed}',
                'warmup_epochs': cfg.warmup, 'main_epochs': 100-cfg.warmup,
                **expected_budget(c), 'best_epoch': 75, 'validation': history[74]['validation'],
                'test': metrics(.2-index*.01), 'final_epoch_validation': history[-1]['validation'],
                'initial_validation': None if c == 'baseline' else metrics(.9),
                'duration_seconds': 700.+index*100+seed, 'peak_gpu_memory_bytes': (index+1)*2**30,
                'final_training_loss': history[-1]['loss'], 'history': history,
                'subset_statistics': dict(zip(('entities','relations','train_triples','valid_triples','test_triples'), COUNTS))})
    return results
