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


def check_gradients(module, label):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), label + ': missing/non-finite gradients'
    assert any(g.abs().sum().item() > 0 for g in grads), label + ': zero gradients'
    print('[ok] finite nonzero gradients reach ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='experiments/checkpoints/kg_only_baseline_rgat_complex.pt')
    parser.add_argument('--lm-name', default='roberta-base')
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
    model = KGTextIntegration(structural, args.lm_name).to(device)
    model.train()
    assert not model.bridge.lm.lm.training
    assert all(not p.requires_grad for p in model.bridge.lm.lm.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    optimizer.zero_grad()
    entities, relations = model(edge_index, edge_type, entity_ids, relation_ids, entity_tokens, relation_tokens)
    dim = cfg['dim']
    assert entities.shape == (len(entity_ids), dim) and relations.shape == (len(relation_ids), dim)
    assert torch.isfinite(entities).all() and torch.isfinite(relations).all()
    # Diagnostic objective verifies connectivity, not link-prediction quality.
    loss = entities.square().mean() + relations.square().mean()
    loss.backward()
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
    print('Phase 3 integration smoke test PASSED')


if __name__ == '__main__':
    main()
