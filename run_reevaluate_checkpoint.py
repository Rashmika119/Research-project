"""Reevaluate saved Phase 5 weights and matched baseline; no training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from evaluation.metrics import evaluate_filtered, build_filter_index
from models.kg_text_refinement import KGTextRefinement
from preprocessing.dataset import load_dataset
from preprocessing.graph_builder import build_train_graph
from preprocessing.entity_text import download_entity_text_files, align_entity_texts
from preprocessing.relation_text import download_relation_text_file, read_relation_text_mapping, align_relation_texts
from training.model_factory import build_model
from run_phase5_training import EnrichedScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Phase 5 best.pt or last.pt')
    parser.add_argument('--baseline', default='experiments/checkpoints/kg_only_baseline_rgat_complex.pt')
    parser.add_argument('--output', help='New JSON report path; defaults beside checkpoint')
    parser.add_argument('--text-batch-size', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix('.tie_report.json')
    if output.exists():
        raise FileExistsError('Report exists; choose a new --output path')
    if args.text_batch_size < 1:
        raise ValueError('Text batch size must be positive')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    baseline = torch.load(args.baseline, map_location='cpu', weights_only=True)
    if baseline['config']['model'] != meta['model_config']:
        raise ValueError('Baseline model configuration differs from experiment')
    full = load_dataset(baseline['config']['dataset']['raw_dir'], download_if_missing=True, use_synthetic_fallback=False)
    for key in ('entity2id', 'relation2id'):
        if key in baseline:
            assert baseline[key] == getattr(full, key), 'Baseline ID mapping mismatch'
        else:
            print('Legacy baseline: assuming unchanged dataset mapping:', key)
    assert meta['relation2id'] == full.relation2id
    entity2id = meta['entity2id']
    ids = [None] * len(entity2id)
    for name, row in entity2id.items():
        ids[row] = full.entity2id[name]
    remap = {original: local for local, original in enumerate(ids)}
    known_train = [(remap[h], r, remap[t]) for h, r, t in full.train if h in remap and t in remap]
    filters = build_filter_index(known_train + meta['valid'] + meta['test'])
    graph = build_train_graph(meta['train'], len(ids), len(meta['relation2id']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    edges = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
    types = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    structural = build_model(meta['model_config'], len(ids), len(meta['relation2id'])).to(device)
    baseline_state = dict(baseline['model_state'])
    baseline_state['entity_emb.weight'] = baseline_state['entity_emb.weight'][ids].clone()
    structural.load_state_dict(baseline_state, strict=True)
    report = {'checkpoint': args.checkpoint, 'epoch': saved['epoch'],
              'ranking': 'average rank of exact ties; optimistic scores included for comparison',
              'warning': 'Saved epoch selected using historical ranking. This does not reselect the best epoch.',
              'candidate_entities': len(ids), 'models': {}}
    def evaluate(label, scorer_model, entities):
        results = {}
        for split in ('valid', 'test'):
            details = []
            metrics = evaluate_filtered(scorer_model, entities, meta[split], *filters, diagnostics=details)
            results[split] = {'metrics': metrics, 'queries': details}
            print(label, split, json.dumps(metrics), flush=True)
        results['representations'] = {'zero_rows': int((entities == 0).all(1).sum()),
            'unique_rows': int(torch.unique(entities, dim=0).shape[0]),
            'total_rows': len(entities)}
        report['models'][label] = results
    structural.eval()
    with torch.no_grad():
        evaluate('baseline', structural, structural.encode(edges, types))
    variant = meta.get('variant', 'original')
    model = KGTextRefinement(structural, lm_name=meta.get('lm_name', 'roberta-base'),
        residual=meta.get('residual', variant != 'original'),
        use_refinement=meta.get('use_refinement', variant != 'no-refinement'),
        refinement_layers=meta.get('refinement_layers', 2),
        refinement_heads=meta.get('refinement_heads', 1)).to(device)
    missing = model.load_state_dict(saved['model_state'], strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('bridge.lm.lm.') for k in missing.missing_keys), missing.missing_keys
    long_path, short_path = download_entity_text_files('data/text/fb15k237')
    descriptions = align_entity_texts(entity2id, long_path, short_path).texts_by_id
    relations = align_relation_texts(meta['relation2id'], read_relation_text_mapping(
        download_relation_text_file('data/text/fb15k237'))).texts_by_id
    tokenizer = AutoTokenizer.from_pretrained(meta.get('lm_name', 'roberta-base'))
    def tokenize(texts):
        tokens = tokenizer([s if s.strip() else '[Missing description]' for s in texts],
            padding=True, truncation=True, max_length=meta.get('max_text_length', 64), return_tensors='pt')
        return {k: v.to(device) for k, v in tokens.items()}
    model.eval()
    with torch.no_grad():
        entities, enriched_relations = model(edges, types, torch.arange(len(ids), device=device),
            torch.arange(len(relations), device=device), tokenize(descriptions), tokenize(relations),
            edges, text_batch_size=args.text_batch_size)
        evaluate(variant, SimpleNamespace(scorer=EnrichedScorer(enriched_relations)), entities)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved tie diagnostics:', output)


if __name__ == '__main__':
    main()
