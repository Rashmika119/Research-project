"""Checkpoint-backed RGAT -> frozen RoBERTa integration on a small train graph.

Default loads the full RGAT + ComplEx checkpoint. --checkpoint can instead
point at the toy checkpoint. Does not overwrite either checkpoint.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.kg_text_integration import KGTextIntegration
from preprocessing.dataset import load_dataset
from preprocessing.entity_text import download_entity_text_files, align_entity_texts
from preprocessing.relation_text import (
    download_relation_text_file, read_relation_text_mapping, align_relation_texts,
)
from preprocessing.toy_subset import make_toy_subset
from training.model_factory import build_model
from models.kg_text_refinement import KGTextRefinement
from models.kg_encoder_rgat import RGATEncoder
from training.losses import bce_loss
from training.negative_sampling import sample_negatives
import random


def check_gradients(module, label):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), label + ': missing/non-finite gradients'
    assert any(g.abs().sum().item() > 0 for g in grads), label + ': zero gradients'
    print('[ok] finite nonzero gradients reach ' + label)


def main(refinement=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='experiments/checkpoints/kg_only_baseline_rgat_complex.pt')
    parser.add_argument('--lm-name', default='roberta-base')
    if refinement:
        parser.add_argument('--variant', choices=['original', 'residual', 'no-refinement'], default='original')
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError('Upload your trained checkpoint to ' + args.checkpoint)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    # Project-produced checkpoints only; no arbitrary third-party pickle files.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    cfg = config['model']
    assert cfg.get('encoder_type') == 'rgat' and cfg.get('scorer_type') == 'complex', 'Requires RGAT + ComplEx checkpoint'
    ds_cfg = config['dataset']
    dataset = load_dataset(raw_dir=ds_cfg['raw_dir'],
                           download_if_missing=ds_cfg.get('download_if_missing', True),
                           use_synthetic_fallback=False)
    if config.get('use_toy_subset', False):
        toy = config['toy_subset']
        dataset = make_toy_subset(dataset, max_entities=toy['max_entities'], seed=toy['seed'])
    assert dataset.num_entities == checkpoint['num_entities']
    assert dataset.num_relations == checkpoint['num_relations']
    for key in ('entity2id', 'relation2id'):
        if key in checkpoint:
            assert checkpoint[key] == getattr(dataset, key), key + ' checkpoint mismatch'
        else:
            print('Legacy checkpoint: reconstructing ' + key + ' from original dataset/config; assumes unchanged data.')
    structural = build_model(cfg, dataset.num_entities, dataset.num_relations)
    structural.load_state_dict(checkpoint['model_state'], strict=True)
    print('[ok] trained RGAT + ComplEx checkpoint loaded')

    # A small induced training graph keeps original global IDs; no remapping
    # or validation/test edges. Full embeddings are loaded, full edges are not.
    selected = set()
    for h, _, t in dataset.train:
        if len(selected | {h, t}) <= 24:
            selected.update((h, t))
        if len(selected) == 24:
            break
    triples = [(h, r, t) for h, r, t in dataset.train if h in selected and t in selected][:64]
    assert triples, 'No training edges selected'
    entity_ids = torch.tensor(sorted({v for h, _, t in triples for v in (h, t)})[:4], device=device)
    relation_ids = torch.tensor(sorted({r for _, r, _ in triples})[:4], device=device)
    edges = [(h, t) for h, _, t in triples] + [(t, h) for h, _, t in triples]
    types = [r for _, r, _ in triples] + [r + dataset.num_relations for _, r, _ in triples]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_type = torch.tensor(types, dtype=torch.long, device=device)
    print('[ok] small train-only graph:', len(triples), 'triples; original IDs retained')
    if refinement:
        entity_ids = torch.tensor(sorted({v for h, _, t in triples for v in (h, t)}), device=device)
        relation_ids = torch.tensor(sorted({r for _, r, _ in triples}), device=device)
        entity_rows = {v: i for i, v in enumerate(entity_ids.tolist())}
        relation_rows = {v: i for i, v in enumerate(relation_ids.tolist())}
        local_edges = torch.tensor([(entity_rows[h], entity_rows[t]) for h, t in edges],
                                   dtype=torch.long, device=device).t().contiguous()
        # Standalone refiner gate before wiring the real semantic outputs.
        refiner = RGATEncoder(cfg['dim'], dataset.num_relations * 2, heads=1).to(device)
        dummy = torch.randn(len(entity_ids), cfg['dim'], device=device, requires_grad=True)
        refined = refiner(dummy, local_edges, edge_type)
        assert refined.shape == dummy.shape and torch.isfinite(refined).all()
        refined.square().mean().backward()
        check_gradients(refiner, 'standalone second RGAT')
        assert dummy.grad is not None and torch.isfinite(dummy.grad).all() and dummy.grad.abs().sum() > 0
        del refiner, dummy, refined

    text_dir = 'data/text/fb15k237'
    long_path, short_path = download_entity_text_files(text_dir)
    entity_text = align_entity_texts(dataset.entity2id, long_path, short_path)
    relation_text = align_relation_texts(dataset.relation2id,
        read_relation_text_mapping(download_relation_text_file(text_dir)))
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    def tokenize(texts, ids):
        descriptions = [texts[i] for i in ids.tolist()]
        assert all(text.strip() for text in descriptions), 'Selected item has no text'
        tokens = tokenizer(descriptions, padding=True, truncation=True, max_length=64, return_tensors='pt')
        return {key: value.to(device) for key, value in tokens.items()}
    entity_tokens = tokenize(entity_text.texts_by_id, entity_ids)
    relation_tokens = tokenize(relation_text.texts_by_id, relation_ids)
    model_class = KGTextRefinement if refinement else KGTextIntegration
    if refinement:
        model = model_class(structural, args.lm_name, residual=args.variant != 'original',
                            use_refinement=args.variant != 'no-refinement').to(device)
        print('Variant:', args.variant)
    else:
        model = model_class(structural, args.lm_name).to(device)
    model.train()
    assert not model.bridge.lm.lm.training
    assert all(not p.requires_grad for p in model.bridge.lm.lm.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    optimizer.zero_grad()
    inputs = (edge_index, edge_type, entity_ids, relation_ids, entity_tokens, relation_tokens)
    entities, relations = model(*inputs, local_edges) if refinement else model(*inputs)
    dim = cfg['dim']
    assert entities.shape == (len(entity_ids), dim) and relations.shape == (len(relation_ids), dim)
    assert torch.isfinite(entities).all() and torch.isfinite(relations).all()
    # Diagnostic objective verifies connectivity, not link-prediction quality.
    loss = entities.square().mean() + relations.square().mean()
    if refinement:
        known = {(entity_rows[h], relation_rows[r], entity_rows[t])
                 for h, r, t in dataset.train
                 if h in entity_rows and t in entity_rows and r in relation_rows}
        positive = torch.tensor([(entity_rows[h], relation_rows[r], entity_rows[t])
                                 for h, r, t in triples[:4]], dtype=torch.long, device=device)
        negative = sample_negatives(positive.cpu(), len(entity_ids), known, 2,
                                    random.Random(0)).to(device)
        positive_scores = model.score_triples(entities, relations, positive)
        negative_scores = model.score_triples(entities, relations, negative.reshape(-1, 3)).reshape(len(positive), 2)
        loss = bce_loss(positive_scores, negative_scores)
        assert torch.isfinite(loss), 'Non-finite full-pipeline BCE loss'
        # Verify explicitly enriched relation scores against complex arithmetic.
        h = torch.complex(*entities[positive[:, 0]].chunk(2, -1))
        r = torch.complex(*relations[positive[:, 1]].chunk(2, -1))
        t = torch.complex(*entities[positive[:, 2]].chunk(2, -1))
        torch.testing.assert_close(positive_scores, (h * r * t.conj()).sum(-1).real)
        print('[ok] enriched-vector ComplEx scores and filtered-negative BCE; loss=', loss.item())
    loss.backward()
    if refinement and model.refiner is not None:
        check_gradients(model.refiner, 'integrated second RGAT')
    if refinement and model.residual:
        gates = [p for name, p in model.named_parameters() if name.endswith('_gate')]
        assert all(p.grad is not None and torch.isfinite(p.grad) for p in gates)
        print('[ok] residual gates receive finite gradients')
    for module, label in ((model.structural.encoder, 'RGAT'),
                          (model.structural.entity_emb, 'entity embeddings'),
                          (model.structural.scorer.relation_emb, 'ComplEx relation embeddings'),
                          (model.bridge.kg_to_lm, 'KG -> LM projection'),
                          (model.bridge.lm_to_kg, 'LM -> KG projection')):
        check_gradients(module, label)
    assert all(p.grad is None for p in model.bridge.lm.lm.parameters())
    print('[ok] frozen RoBERTa has no parameter gradients')
    trainable = next(model.bridge.kg_to_lm.parameters())
    previous = trainable.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    assert not torch.equal(previous, trainable.detach()), 'Optimizer did not update projection'
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print('[ok] optimizer step updates trainable parameters; checkpoint unchanged')
    print('Phase 4 refinement smoke test PASSED' if refinement else 'Phase 3 integration smoke test PASSED')


if __name__ == '__main__':
    main()
