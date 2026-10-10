"""Command-line interface shared by independent baseline and scratch variants."""
import argparse

from training.research_pipeline import ExperimentConfig, run_experiment
from training.variants import VARIANTS, CONFIGURATIONS, configuration_settings


def main(baseline=False):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--variant', '--architecture', dest='variant', choices=VARIANTS)
    selection.add_argument('--configuration', choices=CONFIGURATIONS)
    parser.add_argument('--max-entities', type=int, default=0 if baseline else 5000,
                        help='0 selects the complete dataset')
    parser.add_argument('--epochs', type=int, default=100 if baseline else 30,
                        help='Total epochs INCLUDING structural warmup')
    parser.add_argument('--warmup-epochs', type=int, default=0 if baseline else 5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-negatives', type=int, default=4)
    parser.add_argument('--train-batch-size', type=int, default=32768 if baseline else 0)
    parser.add_argument('--text-batch-size', type=int, default=4)
    parser.add_argument('--max-text-length', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=5 if baseline else 1)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lm-name', default='roberta-base')
    parser.add_argument('--raw-dir', default='data/raw/fb15k237')
    parser.add_argument('--text-dir', default='data/text/fb15k237')
    parser.add_argument('--output-dir', default='experiments/research/baseline' if baseline else 'experiments/research/single')
    parser.add_argument('--manifest')
    parser.add_argument('--cache-dir', default='data/cache/pooled_text')
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    architecture, warmup = args.variant or 'residual', args.warmup_epochs
    if args.configuration:
        if baseline:
            parser.error('Baseline does not accept ablation configurations')
        architecture, warmup = configuration_settings(args.configuration, args.warmup_epochs)
    cfg = ExperimentConfig(variant='baseline' if baseline else architecture,
        max_entities=args.max_entities, epochs=args.epochs, warmup=0 if baseline else warmup,
        seed=args.seed, subset_seed=args.subset_seed, lr=args.lr, num_negatives=args.num_negatives,
        train_batch_size=args.train_batch_size, text_batch_size=args.text_batch_size,
        max_text_length=args.max_text_length, eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size, lm_name=args.lm_name,
        raw_dir=args.raw_dir, text_dir=args.text_dir)
    run_experiment(cfg, args.output_dir, manifest_path=args.manifest, cache_dir=args.cache_dir,
                   skip_completed=args.skip_completed, resume=args.resume)
