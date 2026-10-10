"""Audit real data without training; compare historical traversal and induced facts."""
import argparse
import json

from preprocessing.dataset import load_dataset
from preprocessing.toy_subset import make_toy_subset
from preprocessing.induced_subset import graph_statistics
from training.experiment_io import prepare_data, validate_fb15k237, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw/fb15k237')
    parser.add_argument('--max-entities', type=int, default=5000)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--manifest')
    parser.add_argument('--report', default='experiments/research/subset_audit_5000.json')
    args = parser.parse_args()
    full = load_dataset(args.raw_dir, download_if_missing=True, use_synthetic_fallback=False)
    validate_fb15k237(full)
    _, _, _, manifest = prepare_data(args.raw_dir, args.max_entities, args.subset_seed, args.manifest, full)
    old = make_toy_subset(full, args.max_entities, args.subset_seed)
    report = {'old_traversal': graph_statistics(old), 'new_induced': manifest['statistics'],
              'dataset_sha256': manifest['dataset_sha256'], 'manifest_sha256': manifest['sha256'],
              'same_selected_entities_as_old': set(old.entity2id) == set(manifest['data']['entity2id']),
              'selection_strategy': manifest['selection_strategy'], 'subset_seed': args.subset_seed,
              'old_1000_traversal': graph_statistics(make_toy_subset(full, 1000, args.subset_seed))}
    write_json(args.report, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
