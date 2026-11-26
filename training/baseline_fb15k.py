"""Full FB15K, existing RGAT + ComplEx baseline, seed 0; settings from root baseline_config.yaml."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from preprocessing.dataset import FB15K_COUNTS
from training.experiment_io import prepare_data
from training.research_pipeline import ExperimentConfig, run_experiment

from training.baseline_config import DEFAULT_CONFIG, load_protocol

COUNTS = FB15K_COUNTS


def validate_config(cfg):
    cfg.validate()
    if (cfg.dataset != 'fb15k' or cfg.variant != 'baseline' or cfg.max_entities != 0
            or cfg.seed != 0 or cfg.subset_seed != 0 or cfg.warmup != 0
            or cfg.lm_name or cfg.text_dir or not cfg.deterministic
            or cfg.selection_policy != 'scheduled_only' or not cfg.record_step_losses
            or cfg.model['dim'] <= 0 or cfg.model['dim'] % 2
            or cfg.model['num_layers'] != 2 or cfg.model['heads'] != 1):
        raise ValueError('Expected full FB15K, seed 0, deterministic two-layer RGAT baseline')
    return cfg


def make_config(raw_dir=None, config_path=DEFAULT_CONFIG):
    settings = dict(load_protocol(config_path)['config'])
    if raw_dir is not None:
        settings['raw_dir'] = str(raw_dir)
    return validate_config(ExperimentConfig(**settings))


def expected_budget(cfg=None):
    cfg = make_config() if cfg is None else cfg
    batch = cfg.train_batch_size or COUNTS[2]
    steps = (COUNTS[2] + batch - 1) // batch
    return {'steps_per_epoch': steps, 'optimizer_steps': steps * cfg.epochs,
            'baseline_optimizer_steps': steps * cfg.epochs, 'structural_optimizer_steps': 0,
            'integrated_optimizer_steps': 0, 'last_batch_size': (COUNTS[2]-1) % batch + 1}


def verify_manifest(manifest):
    actual = tuple(manifest['statistics'][k] for k in
                   ('entities', 'relations', 'train_triples', 'valid_triples', 'test_triples'))
    if (actual != COUNTS or manifest.get('dataset') != 'fb15k'
            or manifest['max_entities'] != 0 or manifest['subset_seed'] != 0
            or manifest['selection_strategy'] != 'full_original_splits'):
        raise ValueError('Expected full original FB15K splits, not FB15k-237 or a subset')


def verify_protocol(protocol, cfg):
    expected = dict(protocol['config'])
    expected['raw_dir'] = cfg.raw_dir
    if asdict(validate_config(ExperimentConfig(**expected))) != asdict(cfg):
        raise ValueError('FB15K YAML/runner settings differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Fresh study root; child is baseline_seed0')
    parser.add_argument('--raw-dir', default=None)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
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
    protocol = load_protocol(args.config)
    cfg = make_config(args.raw_dir, args.config)
    root.mkdir(parents=True, exist_ok=True)
    # Save the fetched configuration alongside the results for audit.
    import yaml
    snapshot = root / 'baseline_config.yaml'
    if snapshot.exists() and yaml.safe_load(snapshot.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Output folder belongs to another YAML configuration; use a new run tag')
    snapshot.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding='utf-8')
    # The common runner handles strict source/settings/runtime identity and resume.
    manifest_path = root/'dataset_manifest.json'
    _, _, _, manifest = prepare_data(cfg.raw_dir, 0, 0, manifest_path, dataset='fb15k')
    verify_manifest(manifest)
    print('Protocol:', protocol['protocol'], '| Dataset:', manifest['statistics'], flush=True)
    print('Budget:', expected_budget(cfg), flush=True)
    from training.reproducibility import numerical_policy
    print('Native JIT disabled:', numerical_policy()['native_jit_disabled'], flush=True)
    result = run_experiment(cfg, root/'baseline_seed0', manifest_path=manifest_path,
                            resume=args.resume, skip_completed=args.skip_completed)
    baseline_report(root, result)
