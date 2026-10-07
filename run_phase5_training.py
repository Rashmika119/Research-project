"""Multi-epoch training and held-out ranking of the full graph/text model.

Default: fixed 200-entity real FB15k-237 subset. Original validation/test
splits are retained. Requires the trained full graph-only checkpoint.
"""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.kg_text_refinement import KGTextRefinement
from preprocessing.dataset import load_dataset
from preprocessing.toy_subset import make_toy_subset
from preprocessing.graph_builder import build_train_graph
from preprocessing.entity_text import download_entity_text_files, align_entity_texts
from preprocessing.relation_text import download_relation_text_file, read_relation_text_mapping, align_relation_texts
from training.model_factory import build_model
from training.losses import bce_loss
from training.negative_sampling import sample_negatives


class EnrichedScorer:
    """Candidate scores using the same enriched relations as training."""
    def __init__(self, relations):
        self.relations = relations

    def score_all_tails(self, entities, h, ids):
        hr, hi = h.chunk(2, -1)
        rr, ri = self.relations[ids].chunk(2, -1)
        er, ei = entities.chunk(2, -1)
        return (hr * rr - hi * ri) @ er.t() + (hr * ri + hi * rr) @ ei.t()

    def score_all_heads(self, entities, ids, t):
        rr, ri = self.relations[ids].chunk(2, -1)
        tr, ti = t.chunk(2, -1)
        er, ei = entities.chunk(2, -1)
        return (rr * tr + ri * ti) @ er.t() + (rr * ti - ri * tr) @ ei.t()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='experiments/checkpoints/kg_only_baseline_rgat_complex.pt')
    parser.add_argument('--max-entities', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--text-batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subset-seed', type=int, default=0,
                        help='Keep subset fixed across initialization seeds')
    parser.add_argument('--variant', choices=['original', 'residual', 'no-refinement'], default='residual')
    parser.add_argument('--output-dir', default='experiments/phase5_small')
    args = parser.parse_args()
    if args.epochs < 1 or args.max_entities < 2 or args.lr <= 0 or args.text_batch_size < 1:
        raise ValueError('Invalid training settings')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('best.pt', 'last.pt', 'results.json')):
        raise FileExistsError('Use a new --output-dir to preserve previous experiment results')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = source['config']
    assert not cfg.get('use_toy_subset', False), 'Use the full trained baseline checkpoint'
    assert cfg['model'].get('encoder_type') == 'rgat' and cfg['model'].get('scorer_type') == 'complex'
    full = load_dataset(cfg['dataset']['raw_dir'], download_if_missing=True, use_synthetic_fallback=False)
    assert full.num_entities == source['num_entities'] and full.num_relations == source['num_relations']
    for key in ('entity2id', 'relation2id'):
        if key in source:
            assert source[key] == getattr(full, key), 'Checkpoint ID mapping mismatch'
        else:
            print('Legacy checkpoint: assuming unchanged original dataset for', key)
    data = make_toy_subset(full, max_entities=args.max_entities, seed=args.subset_seed)
    if not data.valid or not data.test:
        raise ValueError('Subset has no validation/test facts. Increase --max-entities; do not invent new splits.')
    print('Subset:', data.num_entities, 'entities;', len(data.train), len(data.valid), len(data.test), 'train/valid/test')
    # Map subset rows to checkpoint rows by entity names, never by row position.
    original_ids = [0] * data.num_entities
    for name, local in data.entity2id.items():
        original_ids[local] = full.entity2id[name]
    state = dict(source['model_state'])
    state['entity_emb.weight'] = state['entity_emb.weight'][original_ids].clone()
    structural = build_model(cfg['model'], data.num_entities, data.num_relations).to(device)
    structural.load_state_dict(state, strict=True)
    graph = build_train_graph(data.train, data.num_entities, data.num_relations)
    edges = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
    types = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    def covered(triples):
        original_to_local = {orig: local for local, orig in enumerate(original_ids)}
        return [(original_to_local[h], r, original_to_local[t]) for h, r, t in triples
                if h in original_to_local and t in original_to_local]
    all_train_known = covered(full.train)
    filters = build_filter_index(all_train_known + data.valid + data.test)
    # Baseline and full model use identical subset graph, candidates and filtering.
    structural.eval()
    with torch.no_grad():
        baseline_entities = structural.encode(edges, types)
        baseline_val = evaluate_filtered(structural, baseline_entities, data.valid, *filters)
        baseline_test = evaluate_filtered(structural, baseline_entities, data.test, *filters)
    print('Matched subset graph-only validation:', baseline_val)
    model = KGTextRefinement(structural, residual=args.variant != 'original',
                             use_refinement=args.variant != 'no-refinement').to(device)
    print('Variant:', args.variant)
    long_path, short_path = download_entity_text_files('data/text/fb15k237')
    texts = align_entity_texts(data.entity2id, long_path, short_path)
    relations = align_relation_texts(data.relation2id, read_relation_text_mapping(
        download_relation_text_file('data/text/fb15k237')))
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    def tokens(descriptions):
        descriptions = [text if text.strip() else '[Missing description]' for text in descriptions]
        batch = tokenizer(descriptions, padding=True, truncation=True, max_length=64, return_tensors='pt')
        return {k: v.to(device) for k, v in batch.items()}
    entity_tokens, relation_tokens = tokens(texts.texts_by_id), tokens(relations.texts_by_id)
    entity_ids = torch.arange(data.num_entities, device=device)
    relation_ids = torch.arange(data.num_relations, device=device)
    def encode():
        return model(edges, types, entity_ids, relation_ids, entity_tokens, relation_tokens,
                     edges, text_batch_size=args.text_batch_size)
    def evaluate(split):
        model.eval()
        with torch.no_grad():
            e, r = encode()
            # Ranking must agree with the explicit-vector scorer used in training.
            example = torch.tensor(split[:1], dtype=torch.long, device=device)
            scorer = EnrichedScorer(r)
            direct = model.score_triples(e, r, example)
            tail_score = scorer.score_all_tails(e, e[example[:, 0]], example[:, 1])[0, example[0, 2]]
            head_score = scorer.score_all_heads(e, example[:, 1], e[example[:, 2]])[0, example[0, 0]]
            torch.testing.assert_close(tail_score, direct[0], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(head_score, direct[0], rtol=1e-4, atol=1e-5)
            return evaluate_filtered(SimpleNamespace(scorer=EnrichedScorer(r)), e, split, *filters)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.00001)
    known = set(all_train_known)
    positives = torch.tensor(data.train, dtype=torch.long, device=device)
    rng = random.Random(args.seed)
    initial_val = evaluate(data.valid)
    print('Untrained full-model validation:', initial_val)
    history, best_mrr, best_epoch = [], initial_val['MRR'], 0
    metadata = {**vars(args), 'lm_name': 'roberta-base', 'max_text_length': 64,
                'num_negatives': 4, 'loss': 'BCE', 'optimizer': 'Adam', 'weight_decay': 0.00001,
                'grad_clip_norm': 1.0, 'model_config': cfg['model'], 'refinement_layers': 2,
                'refinement_heads': 1, 'residual': args.variant != 'original',
                'use_refinement': args.variant != 'no-refinement',
                'entity2id': data.entity2id, 'relation2id': data.relation2id,
                'train': data.train, 'valid': data.valid, 'test': data.test,
                'baseline_validation': baseline_val, 'baseline_test': baseline_test,
                'initial_validation': initial_val}
    def save(path, epoch):
        # Frozen pretrained LM can be reconstructed; save all trainable weights.
        torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()
                                    if not k.startswith('bridge.lm.lm.')},
                    'optimizer_state': optimizer.state_dict(), 'metadata': metadata,
                    'epoch': epoch, 'history': history, 'best_epoch': best_epoch,
                    'best_val_mrr': best_mrr}, path)
    # Retain epoch zero if training never improves validation.
    save(output / 'best.pt', 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        negative = sample_negatives(positives.cpu(), data.num_entities, known, 4, rng).to(device)
        e, r = encode()
        pos = model.score_triples(e, r, positives)
        neg = model.score_triples(e, r, negative.reshape(-1, 3)).reshape(len(positives), 4)
        loss = bce_loss(pos, neg)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert all(p.grad is None for p in model.bridge.lm.lm.parameters())
        optimizer.step()
        loss_value = loss.item()
        del e, r, pos, neg, loss
        val = evaluate(data.valid)
        history.append({'epoch': epoch, 'loss': loss_value, 'validation': val})
        print(f'Epoch {epoch:03d} | loss={loss_value:.4f} | val_MRR={val["MRR"]:.4f} | val_Hits@10={val["Hits@10"]:.4f}', flush=True)
        if val['MRR'] > best_mrr:
            best_mrr, best_epoch = val['MRR'], epoch
            save(output / 'best.pt', epoch)
        save(output / 'last.pt', epoch)
    chosen = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    incompatible = model.load_state_dict(chosen['model_state'], strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('bridge.lm.lm.') for k in incompatible.missing_keys)
    final_val, final_test = evaluate(data.valid), evaluate(data.test)
    results = {'settings': metadata, 'history': history, 'best_epoch': best_epoch,
               'full_model_validation': final_val, 'full_model_test': final_test,
               'learned_gates': {name: float(param.detach().tanh().cpu())
                                 for name, param in model.named_parameters() if name.endswith('_gate')},
               'note': 'Small subset with restricted candidates, not full FB15k-237 benchmark scores. Baseline initialized from full training facts.'}
    (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('Best epoch:', best_epoch)
    print('Full model validation:', final_val)
    print('Full model held-out test:', final_test)
    print('Matched graph-only held-out test:', baseline_test)
    print('Results and checkpoints:', output)


if __name__ == '__main__':
    main()
