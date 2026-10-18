"""Full FB15K, existing RGAT + ComplEx baseline, seed 0, 300 epochs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from preprocessing.dataset import FB15K_COUNTS
from training.experiment_io import prepare_data
from training.research_pipeline import ExperimentConfig, run_experiment
from training.variants import MODEL_CONFIG

PROTOCOL = 'fb15k_rgat_complex_dim100_neg10_300_seed0_accum8_v2'
COUNTS = FB15K_COUNTS


def make_config(raw_dir='data/raw/fb15k'):
    cfg = ExperimentConfig(dataset='fb15k', variant='baseline', raw_dir=str(raw_dir),
        lm_name='', text_dir='',
        max_entities=0, seed=0, subset_seed=0, epochs=300, warmup=0,
        model={**MODEL_CONFIG, 'dim': 100}, num_negatives=10,
        train_batch_size=32768, eval_every=10, eval_batch_size=512,
        micro_batch_size=4096, accumulation_steps=8,
        lr=.001, weight_decay=.00001, grad_clip=1., deterministic=True,
        selection_policy='scheduled_only', record_step_losses=True)
    cfg.validate()
    return cfg


def expected_budget():
    return {'steps_per_epoch': 15, 'optimizer_steps': 4500,
            'baseline_optimizer_steps': 4500, 'structural_optimizer_steps': 0,
            'integrated_optimizer_steps': 0, 'last_batch_size': 24390}


def verify_manifest(manifest):
    actual = tuple(manifest['statistics'][k] for k in
                   ('entities', 'relations', 'train_triples', 'valid_triples', 'test_triples'))
    if (actual != COUNTS or manifest.get('dataset') != 'fb15k'
            or manifest['max_entities'] != 0 or manifest['subset_seed'] != 0
            or manifest['selection_strategy'] != 'full_original_splits'):
        raise ValueError('Expected full original FB15K splits, not FB15k-237 or a subset')


def verify_protocol(protocol, cfg):
    if protocol['protocol'] != PROTOCOL or protocol['config'] != asdict(cfg):
        raise ValueError('FB15K notebook/YAML/runner settings differ')
    if protocol['budget'] != expected_budget():
        raise ValueError('Incorrect FB15K optimizer budget')
    counts = tuple(protocol['counts'][k] for k in
                   ('entities', 'relations', 'train_triples', 'valid_triples', 'test_triples'))
    expected = {'optimizer':'Adam', 'precision':'float32', 'early_stopping':False,
                'scheduler':None, 'gradient_accumulation':True, 'activation_checkpointing':False}
    if counts != COUNTS or any(protocol.get(k) != v for k,v in expected.items()):
        raise ValueError('Incorrect dataset or numerical protocol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Fresh study root; child is baseline_seed0')
    parser.add_argument('--raw-dir', default='data/raw/fb15k')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir)
    from training.baseline_fb15k_reports import baseline_report
    if args.report_only:
        result = json.loads((root/'baseline_seed0/results.json').read_text(encoding='utf-8'))
        baseline_report(root, result)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('FB15K research training requires CUDA; no training was started')
    cfg = make_config(args.raw_dir)
    # The common runner handles strict source/settings/runtime identity and resume.
    manifest_path = root/'dataset_manifest.json'
    _, _, _, manifest = prepare_data(cfg.raw_dir, 0, 0, manifest_path, dataset='fb15k')
    verify_manifest(manifest)
    print('Protocol:', PROTOCOL, '| Dataset:', manifest['statistics'], flush=True)
    print('Budget:', expected_budget(), flush=True)
    result = run_experiment(cfg, root/'baseline_seed0', manifest_path=manifest_path,
                            resume=args.resume, skip_completed=args.skip_completed)
    baseline_report(root, result)
