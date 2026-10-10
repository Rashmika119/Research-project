"""Offline tests by default; --real-lm checks all five variants with RoBERTa."""
import argparse
import os
import subprocess
import sys


def real_lm_smoke(requested_device='auto'):
    import gc
    import torch
    from models.pretrained import load_pretrained_tokenizer
    from run_pretrained_check import select_device
    from preprocessing.dataset import load_synthetic_toy_graph
    from preprocessing.graph_builder import build_train_graph
    from training.research_pipeline import seed_everything, state_fingerprint
    from training.variants import VARIANTS, build_scratch_model

    device = select_device(requested_device)
    data = load_synthetic_toy_graph(8, 2, 16, 3, 3, seed=4)
    graph = build_train_graph(data.train, data.num_entities, data.num_relations)
    edges = torch.tensor(graph.edge_index, device=device).t().contiguous()
    types = torch.tensor(graph.edge_type, device=device)
    entity_ids = torch.arange(data.num_entities, device=device)
    relation_ids = torch.arange(data.num_relations, device=device)
    tokenizer = load_pretrained_tokenizer()
    def tokens(count, label):
        encoded = tokenizer([f'{label} {i} has an example description.' for i in range(count)],
                            padding=True, truncation=True, max_length=16, return_tensors='pt')
        return {key: encoded[key] for key in ('input_ids', 'attention_mask')}
    et, rt = tokens(data.num_entities, 'Entity'), tokens(data.num_relations, 'Relation')
    positives = torch.tensor(data.train, device=device)
    print('Real frozen RoBERTa smoke tests on', device, '(synthetic facts; no research metrics)', flush=True)
    for variant in VARIANTS:
        seed_everything(0)
        model = build_scratch_model(variant, data.num_entities, data.num_relations).to(device)
        assert model.soft_prompt == (variant != 'residual-no-softprompt')
        if not model.soft_prompt:
            assert not hasattr(model.bridge, 'kg_to_lm')
        before = state_fingerprint(model.bridge.lm)
        model.train()
        e, r = model(edges, types, entity_ids, relation_ids, et, rt, edges, text_batch_size=2)
        scores = model.score_triples(e, r, positives)
        loss = torch.nn.functional.softplus(-scores).mean() + .01 * (e.square().mean() + r.square().mean())
        loss.backward()
        modules = {'first_RGAT': model.structural.encoder, 'entities': model.structural.entity_emb,
                   'relations': model.structural.scorer.relation_emb, 'LM_to_KG': model.bridge.lm_to_kg}
        if model.soft_prompt:
            modules['KG_to_LM'] = model.bridge.kg_to_lm
        if model.refiner is not None:
            modules['second_RGAT'] = model.refiner
        for label, module in modules.items():
            grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), (variant, label)
            assert any(g.abs().sum() > 0 for g in grads), (variant, label, 'zero gradients')
        for name, param in model.named_parameters():
            if name.endswith('_gate'):
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert all(p.grad is None for p in model.bridge.lm.parameters())
        torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001).step()
        assert before == state_fingerprint(model.bridge.lm)
        print(variant, 'forward/backward and frozen-LM check PASSED', flush=True)
        del model, modules, module, param, grads, e, r, scores, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--real-lm', '--real-lm-only', dest='real_lm', action='store_true')
    group.add_argument('--offline-only', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.real_lm:
        real_lm_smoke(args.device)
    else:
        environment = os.environ.copy()
        if args.device in ('auto', 'cpu'):
            environment['CUDA_VISIBLE_DEVICES'] = ''
        else:
            from run_pretrained_check import select_device
            select_device('cuda')
        print('Offline regression suite; device=' + ('cuda' if args.device == 'cuda' else 'cpu'), flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], env=environment)
        if result.returncode:
            print('Offline regressions failed. See the test names and tracebacks above.', file=sys.stderr)
        sys.exit(result.returncode)
