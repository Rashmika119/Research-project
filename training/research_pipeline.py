"""Independent baseline and scratch graph/text experiments with persistent state.

Warmup is structural BCE training at the SAME learning rate, included in the
total epoch budget. All modules are initialized before warmup. Only integrated
validation scores select a variant checkpoint; warmup scores are diagnostic.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from models.pretrained import prepare_pretrained, load_pretrained_tokenizer

from evaluation.enriched_scorer import EnrichedScorer
from evaluation.metrics import build_filter_index, evaluate_filtered
from models.scorer import ComplExScorer
from preprocessing.entity_text import download_entity_text_files, align_entity_texts
from preprocessing.relation_text import (download_relation_text_file,
                                        read_relation_text_mapping, align_relation_texts)
from training.experiment_io import (digest, environment, file_digest, prepare_data,
                                    save_checkpoint, write_json)
from training.losses import bce_loss
from training.negative_sampling import sample_negatives
from training.text_cache import lm_identity, pooled_text
from training.variants import MODEL_CONFIG, VARIANTS, build_scratch_model, warmup_epochs, configuration_id
from training.reproducibility import (seed_everything, capture_rng, numerical_policy,
                                      restore_training_state)


@dataclass
class ExperimentConfig:
    variant: str = 'residual'
    max_entities: int = 5000  # zero means the complete original dataset
    epochs: int = 30  # INCLUDES warmup
    warmup: int = 5
    seed: int = 0
    subset_seed: int = 0
    lr: float = 0.001
    weight_decay: float = 0.00001
    num_negatives: int = 4
    grad_clip: float = 1.0
    train_batch_size: int = 0  # zero = one optimizer step per epoch
    text_batch_size: int = 4
    max_text_length: int = 64
    eval_every: int = 1
    eval_batch_size: int = 64
    lm_name: str = 'roberta-base'
    raw_dir: str = 'data/raw/fb15k237'
    text_dir: str = 'data/text/fb15k237'
    model: dict = field(default_factory=lambda: dict(MODEL_CONFIG))
    deterministic: bool = True
    selection_policy: str = 'initial_and_scheduled'  # historical default
    record_step_losses: bool = False

    def validate(self):
        if self.selection_policy not in ('initial_and_scheduled', 'scheduled_only'):
            raise ValueError('Unknown checkpoint-selection policy')
        if self.variant not in (*VARIANTS, 'baseline'):
            raise ValueError('Unknown architecture')
        if (self.max_entities < 0 or self.max_entities == 1 or self.lr <= 0
                or self.weight_decay < 0 or self.num_negatives < 1
                or self.grad_clip <= 0 or self.train_batch_size < 0
                or self.text_batch_size < 1 or self.max_text_length < 2
                or self.eval_every < 1 or self.eval_batch_size < 1):
            raise ValueError('Invalid experiment settings')
        if self.model.get('encoder_type') != 'rgat' or self.model.get('scorer_type') != 'complex':
            raise ValueError('This protocol requires RGAT + ComplEx')
        warmup_epochs(self.variant, self.epochs, self.warmup)


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.startswith('bridge.lm.lm.')}


def load_model_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not k.startswith('bridge.lm.lm.')
                                          for k in incompatible.missing_keys):
        raise ValueError(f'Checkpoint does not match architecture: {incompatible}')


def text_inputs(data, cfg, revision=None):
    long_path, short_path = download_entity_text_files(cfg.text_dir)
    entities = align_entity_texts(data.entity2id, long_path, short_path).texts_by_id
    relations = align_relation_texts(data.relation2id, read_relation_text_mapping(
        download_relation_text_file(cfg.text_dir))).texts_by_id
    tokenizer = load_pretrained_tokenizer(cfg.lm_name, revision=revision)
    def tokenize(texts):
        encoded = tokenizer([s if s.strip() else '[Missing description]' for s in texts],
                            padding=True, truncation=True, max_length=cfg.max_text_length,
                            return_tensors='pt')
        return {k: encoded[k].cpu() for k in ('input_ids', 'attention_mask')}
    return tokenize(entities), tokenize(relations)


def token_identity(tokens):
    return digest([{k: v.tolist() for k, v in group.items()} for group in tokens])


def validation_due(cfg, epoch, warm):
    return epoch % cfg.eval_every == 0 or epoch in (warm, cfg.epochs)


def can_select_initial(cfg):
    return cfg.selection_policy == 'initial_and_scheduled'


def completed_result(output, run_id):
    output = Path(output)
    path = output / 'results.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('run_id') != run_id or result.get('status') != 'complete':
        raise ValueError('Existing results use different settings/data/code/environment; use a new directory')
    for name, checksum in result['checkpoint_sha256'].items():
        if not (output / name).is_file() or file_digest(output / name) != checksum:
            raise ValueError('Completed-run checkpoint is missing or changed: ' + name)
    return result


def run_experiment(cfg, output_dir, *, manifest_path=None, cache_dir=None,
                   skip_completed=False, resume=False, full_dataset=None, tokens=None):
    """Run without accepting any baseline or pilot initialization checkpoint.

    Resume loads ONLY last.pt from this run after verifying its complete identity.
    full_dataset/tokens injection supports lightweight offline tests.
    """
    cfg.validate()
    data, graph, known_train, manifest = prepare_data(
        cfg.raw_dir, cfg.max_entities, cfg.subset_seed, manifest_path, full_dataset)
    provenance = environment()
    seed_everything(cfg.seed, cfg.deterministic)
    identity = {'protocol': 'scratch_v2_complete_rng', 'config': asdict(cfg),
                'numerical_policy': numerical_policy(),
                'manifest_sha256': manifest['sha256'],
                'source_sha256': provenance['source_sha256'],
                'packages': provenance['packages'], 'device': provenance['device'],
                'python': provenance['python'], 'cuda': provenance['cuda']}
    # Text provenance is part of completed-run validation, even when no training
    # is needed. Reading cached tokenizer/text resources avoids another LM load.
    lm_revision = None
    if cfg.variant != 'baseline':
        if tokens is None:
            lm_revision = prepare_pretrained(cfg.lm_name).revision
            identity['lm_revision'] = lm_revision
            if lm_revision is None:
                local_lm = Path(cfg.lm_name)
                if not local_lm.is_dir():
                    raise ValueError('Frozen LM must resolve to a Hub revision or a local model directory')
                identity['local_lm_files'] = {str(p.relative_to(local_lm)): file_digest(p)
                                             for p in sorted(local_lm.rglob('*')) if p.is_file()}
            tokens = text_inputs(data, cfg, lm_revision)
        identity['tokens_sha256'] = token_identity(tokens)
    run_id = digest(identity)
    output = Path(output_dir)
    if (output / 'results.json').exists():
        result = completed_result(output, run_id)
        if skip_completed:
            print('Verified completed run; skipping:', output, flush=True)
            return result
        raise FileExistsError('Completed run exists; use --skip-completed or a new output directory')
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError('Partial run exists; use --resume or a new output directory')
        if not (output / 'config.json').is_file():
            raise FileExistsError('Cannot resume an unrecognized nonempty directory')
    output.mkdir(parents=True, exist_ok=True)
    if resume and (output / 'config.json').exists():
        saved_config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if saved_config['run_id'] != run_id:
            raise ValueError('Resume settings/data/source/environment differ from this run')
    write_json(output / 'config.json', {'run_id': run_id, **identity, 'environment': provenance})
    write_json(output / 'manifest.json', manifest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed, cfg.deterministic)
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    identifier = configuration_id(cfg.variant, cfg.warmup)
    print(f'Device: {device}; configuration={identifier}; architecture={cfg.variant}; '
          f'warmup={cfg.warmup}; seed={cfg.seed}; initialization=scratch', flush=True)
    print(f'Data: {data.num_entities} entities, {len(data.train)}/{len(data.valid)}/{len(data.test)} train/valid/test', flush=True)
    # Factory has no checkpoint input. This is the ONLY initial model creation.
    model = build_scratch_model(cfg.variant, data.num_entities, data.num_relations,
                                cfg.model, cfg.lm_name, lm_revision)
    # Keep independent frozen text on CPU on cache hits. Moving the rest of
    # the model preserves every registered parameter/buffer and its identity.
    if cfg.variant == 'residual-no-softprompt':
        frozen_lm = model.bridge.lm
        model.bridge.lm = None
        try:
            model.to(device)
        finally:
            model.bridge.lm = frozen_lm
    else:
        model.to(device)
    structural = model if cfg.variant == 'baseline' else model.structural
    initial_structural_sha = state_fingerprint(structural)
    initial_trainable_sha = state_dict_fingerprint(trainable_state(model))
    lm_info = None if cfg.variant == 'baseline' else lm_identity(model.bridge.lm)
    entity_pooled = relation_pooled = None
    cache_keys = None
    if cfg.variant == 'residual-no-softprompt':
        entity_pooled, ek = pooled_text(model.bridge.lm, tokens[0], cfg.text_batch_size, device, cache_dir)
        relation_pooled, rk = pooled_text(model.bridge.lm, tokens[1], cfg.text_batch_size, device, cache_dir)
        cache_keys = {'entities': ek, 'relations': rk}
        # No later forward needs the LM in this variant. Release its GPU space
        # and keep the much smaller pooled matrices resident for training.
        model.bridge.lm.to('cpu')
        entity_pooled, relation_pooled = entity_pooled.to(device), relation_pooled.to(device)
    elif cfg.variant != 'baseline':
        tokens = tuple({k: v.to(device) for k, v in group.items()} for group in tokens)
    # Keep construction/download randomness outside the training RNG stream.
    seed_everything(cfg.seed + 20000, cfg.deterministic)
    negative_rng = random.Random(cfg.seed)
    shuffle_rng = torch.Generator().manual_seed(cfg.seed)
    edges = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
    types = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    entity_ids = torch.arange(data.num_entities, device=device)
    relation_ids = torch.arange(data.num_relations, device=device)
    positives_cpu = torch.tensor(data.train, dtype=torch.long)
    known = set(known_train)
    filters = build_filter_index(known_train + data.valid + data.test)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = warmup_epochs(cfg.variant, cfg.epochs, cfg.warmup)
    batch_size = cfg.train_batch_size or len(data.train)
    steps_per_epoch = math.ceil(len(data.train) / batch_size)
    print(f'Budget: {warm} structural + {cfg.epochs - warm} main epochs; '
          f'{cfg.epochs * steps_per_epoch} optimizer steps; fixed lr={cfg.lr}', flush=True)
    history, best_mrr, best_epoch, start_epoch, optimizer_steps = [], -1., -1, 1, 0
    best_state = None
    initial_validation, elapsed_before, previous_peak = None, 0., 0

    def encode(structural_only=False):
        if structural_only or cfg.variant == 'baseline':
            return structural.encode(edges, types), structural.scorer.relation_emb.weight
        return model(edges, types, entity_ids, relation_ids, tokens[0], tokens[1], edges,
                     text_batch_size=cfg.text_batch_size,
                     entity_pooled=entity_pooled, relation_pooled=relation_pooled)

    def evaluate(split, structural_only=False):
        model.eval()
        with torch.no_grad():
            e, r = encode(structural_only)
            scorer = EnrichedScorer(r)
            h, rel, t = split[0]
            direct = ComplExScorer.score_vectors(e[h:h+1], r[rel:rel+1], e[t:t+1])[0]
            ids = torch.tensor([rel], device=device)
            torch.testing.assert_close(scorer.score_all_tails(e, e[h:h+1], ids)[0, t], direct,
                                       rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(scorer.score_all_heads(e, ids, e[t:t+1])[0, h], direct,
                                       rtol=1e-4, atol=1e-5)
            return evaluate_filtered(SimpleNamespace(scorer=scorer), e, split, *filters,
                                     batch_size=cfg.eval_batch_size)

    def peak_memory():
        return max(previous_peak, torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)

    def save(name, epoch, selected_state=None):
        nonlocal best_state
        current_state = selected_state if selected_state is not None else trainable_state(model)
        if name == 'best.pt':
            best_state = current_state
        # Baseline fields remain readable by existing baseline inspection tools.
        legacy_cfg = {'model': cfg.model, 'seed': cfg.seed,
                      'dataset': {'raw_dir': cfg.raw_dir, 'download_if_missing': True,
                                  'use_synthetic_fallback': False},
                      'use_toy_subset': cfg.max_entities != 0,
                      'toy_subset': {'max_entities': cfg.max_entities, 'seed': cfg.subset_seed}}
        save_checkpoint(output / name, {
            'run_id': run_id, 'model_state': current_state,
            'best_model_state': best_state if name == 'last.pt' else None,
            'optimizer_state': optimizer.state_dict(), 'config': legacy_cfg,
            'experiment_config': asdict(cfg), 'num_entities': data.num_entities,
            'num_relations': data.num_relations, 'entity2id': data.entity2id,
            'relation2id': data.relation2id, 'epoch': epoch, 'history': history,
            'best_epoch': best_epoch, 'best_val_mrr': best_mrr,
            'optimizer_steps': optimizer_steps, 'initial_validation': initial_validation,
            'initial_structural_sha256': initial_structural_sha, 'lm_identity': lm_info,
            'initial_trainable_sha256': initial_trainable_sha,
            'elapsed_seconds': elapsed_before + time.perf_counter() - started,
            'peak_gpu_memory_bytes': peak_memory(),
            'checkpoint_boundary': 'epoch_end' if name == 'last.pt' else 'selection_only',
            'module_modes': {n: m.training for n, m in model.named_modules()},
            'scheduler_state': None, 'grad_scaler_state': None,
            'numerical_policy': numerical_policy(),
            'rng': capture_rng(negative_rng, shuffle_rng),
        })

    if resume and (output / 'last.pt').exists():
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if saved['run_id'] != run_id or saved['lm_identity'] != lm_info:
            raise ValueError('Resume checkpoint identity or frozen LM revision changed')
        audit = restore_training_state(model, optimizer, saved, negative_rng, shuffle_rng)
        write_json(output / 'resume_audit.json', audit)
        history, best_mrr, best_epoch = saved['history'], saved['best_val_mrr'], saved['best_epoch']
        start_epoch, optimizer_steps = saved['epoch'] + 1, saved['optimizer_steps']
        initial_validation = saved['initial_validation']
        initial_structural_sha = saved['initial_structural_sha256']
        initial_trainable_sha = saved.get('initial_trainable_sha256', initial_trainable_sha)
        best_state = saved['best_model_state']
        elapsed_before, previous_peak = saved['elapsed_seconds'], saved['peak_gpu_memory_bytes']
        # last.pt is the epoch commit. It carries the selected weights so a
        # disconnect between the two atomic checkpoint writes is recoverable.
        if best_state is not None:
            save('best.pt', best_epoch, selected_state=best_state)
        print('Resuming this run at epoch', start_epoch, flush=True)

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            structural_only = epoch <= warm
            if cfg.variant != 'baseline' and not structural_only and initial_validation is None:
                initial_validation = evaluate(data.valid)
                if can_select_initial(cfg):
                    best_mrr, best_epoch = initial_validation['MRR'], epoch - 1
                    save('best.pt', best_epoch)
                print('Initial integrated validation (selection policy: '
                      + cfg.selection_policy + '):', initial_validation, flush=True)
            epoch_started = time.perf_counter()
            model.train()
            permutation = torch.randperm(len(positives_cpu), generator=shuffle_rng)
            total_loss, steps = 0., 0
            step_losses = []
            weighted_loss = 0.
            for offset in range(0, len(permutation), batch_size):
                positive_cpu = positives_cpu[permutation[offset:offset + batch_size]]
                positive = positive_cpu.to(device)
                negative = sample_negatives(positive_cpu, data.num_entities, known,
                                            cfg.num_negatives, negative_rng).to(device)
                optimizer.zero_grad(set_to_none=True)
                e, r = encode(structural_only)
                def score(triples):
                    return ComplExScorer.score_vectors(e[triples[:, 0]], r[triples[:, 1]], e[triples[:, 2]])
                loss = bce_loss(score(positive), score(negative.reshape(-1, 3)).reshape(
                    len(positive), cfg.num_negatives))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                if cfg.variant != 'baseline' and any(p.grad is not None for p in model.bridge.lm.parameters()):
                    raise AssertionError('Frozen LM received parameter gradients')
                optimizer.step()
                loss_value = loss.item()
                total_loss += loss_value
                weighted_loss += loss_value * len(positive_cpu)
                optimizer_steps += 1
                steps += 1
                if cfg.record_step_losses:
                    step_losses.append({'optimizer_step': optimizer_steps,
                                        'positive_count': len(positive_cpu), 'loss': loss_value})
                del e, r, loss, positive, negative
            row = {'epoch': epoch, 'phase': 'structural_warmup' if structural_only else 'main',
                   'loss': total_loss / steps, 'optimizer_steps': optimizer_steps,
                   'train_seconds': time.perf_counter() - epoch_started}
            if cfg.record_step_losses:
                row.update(step_losses=step_losses,
                           example_weighted_loss=weighted_loss / len(positives_cpu))
            if validation_due(cfg, epoch, warm):
                val = evaluate(data.valid, structural_only)
                row['validation'] = val
                row['validation_model'] = ('baseline' if cfg.variant == 'baseline' else
                                           'structural' if structural_only else 'integrated')
            row['epoch_seconds'] = time.perf_counter() - epoch_started
            row['elapsed_seconds'] = elapsed_before + time.perf_counter() - started
            row['gpu_memory_bytes'] = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
            row['peak_gpu_memory_bytes'] = peak_memory() if device.type == 'cuda' else None
            history.append(row)
            print(f"Epoch {epoch:03d}/{cfg.epochs} | {row['phase']} | loss={row['loss']:.6f}"
                  + (f" | val_MRR={row['validation']['MRR']:.6f}"
                     f" | val_Hits@10={row['validation']['Hits@10']:.6f}" if 'validation' in row else '')
                  + f" | GPU_peak={row['peak_gpu_memory_bytes']} bytes | elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            if not structural_only and 'validation' in row and row['validation']['MRR'] > best_mrr:
                best_mrr, best_epoch = row['validation']['MRR'], epoch
                save('best.pt', epoch)
            save('last.pt', epoch)
            write_json(output / 'history.json', history)
            if cfg.record_step_losses:
                from training.reports import write_csv
                write_csv(output / 'steps.csv', [dict(epoch=h['epoch'], phase=h['phase'], **s)
                          for h in history for s in h.get('step_losses', [])])
        chosen = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        if chosen['run_id'] != run_id:
            raise ValueError('Best checkpoint belongs to a different run')
        load_model_state(model, chosen['model_state'])
        validation, test = evaluate(data.valid), evaluate(data.test)
    except torch.cuda.OutOfMemoryError:
        write_json(output / 'failure.json', {'reason': 'CUDA out of memory', 'run_id': run_id,
                   'advice': 'Use a larger GPU or a new run with smaller text/evaluation batches. '
                             'Full-graph RGAT memory also depends on graph size; text batching cannot fix that.'})
        raise
    result = {'status': 'complete', 'run_id': run_id, 'settings': asdict(cfg),
              'configuration': identifier, 'architecture': cfg.variant,
              'initialization': 'scratch', 'baseline_checkpoint_used': False,
              'dataset_scope': 'full' if cfg.max_entities == 0 else 'subset',
              'manifest_sha256': manifest['sha256'], 'dataset_sha256': manifest['dataset_sha256'],
              'candidate_entities': data.num_entities, 'ranking_policy': 'average_exact_ties',
              'warmup_epochs': warm, 'main_epochs': cfg.epochs - warm,
              'optimizer_steps': optimizer_steps, 'steps_per_epoch': steps_per_epoch,
              'structural_optimizer_steps': warm * steps_per_epoch,
              'integrated_optimizer_steps': (cfg.epochs - warm) * steps_per_epoch if cfg.variant != 'baseline' else 0,
              'baseline_optimizer_steps': optimizer_steps if cfg.variant == 'baseline' else 0,
              'best_epoch': best_epoch, 'best_main_epoch': best_epoch - warm,
              'validation': validation, 'test': test, 'initial_validation': initial_validation,
              'initial_validation_epoch': warm if cfg.variant != 'baseline' else None,
              'final_epoch_validation': history[-1].get('validation'),
              'final_training_loss': history[-1]['loss'],
              'subset_statistics': manifest['statistics'],
              'history': history, 'duration_seconds': elapsed_before + time.perf_counter() - started,
              'peak_gpu_memory_bytes': peak_memory() if device.type == 'cuda' else None,
              'initial_structural_sha256': initial_structural_sha, 'lm_identity': lm_info,
              'initial_trainable_sha256': initial_trainable_sha,
              'text_cache_keys': cache_keys, 'environment': provenance,
              'learned_gates': {n: float(p.detach().tanh().cpu()) for n, p in model.named_parameters()
                                if n.endswith('_gate')},
              'checkpoint_sha256': {name: file_digest(output / name) for name in ('best.pt', 'last.pt')},
              'numerical_policy': numerical_policy(),
              'reproducibility_note': 'Strict deterministic algorithms by default; unsupported kernels fail loudly. '
                                      'Reproducibility is limited to matching devices, libraries and numerical policy.'}
    from training.reports import export_run
    write_json(output / 'history.json', history)
    export_run(output, result)
    # Completion marker is last, after all per-run tabular artifacts exist.
    write_json(output / 'results.json', result)
    print('Best epoch:', best_epoch, '| validation:', validation, '| test:', test, flush=True)
    print('Saved:', output, flush=True)
    return result


def state_fingerprint(model):
    return state_dict_fingerprint(model.state_dict())


def state_dict_fingerprint(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
