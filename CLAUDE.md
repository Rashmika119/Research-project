# CLAUDE.md

Implementation guidance for this research repository. Read [README.md](README.md)
for the architecture and commands, and [experiments/RUNS.md](experiments/RUNS.md)
for historical results, provenance and checkpoint recovery.

## Current status

Updated 2026-10-05 (Asia/Colombo).

**Phase 3 is implemented and offline-tested; the real-checkpoint / pretrained-LM
gate is pending.** Phases 0?2 have reported historical passes. Twelve offline
Phase 3 tests pass, but they do not establish a real FB15k-237 gate pass or
improved link prediction. `kg_only_baseline.pt` is not present locally.

Phase 3 uses **R-GCN, dimension 128, two layers, 30 bases, dropout 0.2**,
matching the reported full baseline. The shared entity/relation bridge uses
frozen `roberta-base` and projection dropout 0.1:

```text
Restored R-GCN entity outputs / learned scorer relation embeddings [128]
  -> KG-to-LM projection [768]
  -> KG soft prompt + aligned text through frozen RoBERTa
  -> contextual prompt representation [768]
  -> LM-to-KG projection [128]
```

[phase3_integration.yaml](experiments/configs/phase3_integration.yaml) is the
current gate config. The historical Phase 2 tests use dimension 32; that is not
an architectural restriction. RGAT remains an optional comparison track, with
its toy gate reported passed and its revised full run unverified. Do not switch
the primary encoder or dimension implicitly.

## Implementation rules

1. **Preserve identity.** Entity and relation text must use the KG's existing ID
   maps. Copy checkpoint rows into toy ID order by original entity key. The toy
   graph retains full relation IDs and inverse-edge offsets. Counts alone cannot
   establish checkpoint compatibility.
2. **Keep held-out triples out of training.** Build message-passing edges and
   negative-sampling filters from training triples only. Use all splits together
   only for known-positive filtering during ranking evaluation.
3. **Freeze the LM without breaking autograd.** Keep all LM parameters frozen
   and its module in eval mode, including after a parent `.train()` call. Do not
   wrap a training-time LM forward in `torch.no_grad()` or detach KG inputs.
4. **Use the existing BCE/logistic objective with DistMult.** Do not revive the
   rejected diagnostic margin-loss setup or initialize KG parameters directly
   from raw LM outputs. Feed semantic information through trainable projections.
5. **Encode the graph once per optimizer step** and reuse its representations
   for that step's positive and negative triples.
6. **Keep the learning rate fixed within a run.** Record hyperparameter changes
   as separate runs so comparisons remain attributable.
7. **Keep FB15k-237 as the primary dataset.** Full-data/gate runs must fail if
   real data is unavailable; synthetic graphs belong only in explicit software
   tests. WN18/WN18RR diagnostic numbers are not comparable baselines.
8. **Do not fabricate recovery or results.** Missing checkpoints, original
   mappings, run revisions or logs must stay marked unknown. Retraining creates
   a new run, not a recovered historical result.

## Phase 3 gate contract

The implementation is [run_phase3_smoke_test.py](run_phase3_smoke_test.py), with
helpers in [training/phase3_gate.py](training/phase3_gate.py).

- Strictly validate checkpoint model settings, state shapes, finite tensors,
  counts and exact ID maps. New Phase 1 saves embed both maps. Legacy saves
  require an original-run mapping sidecar bound to the checkpoint's SHA-256;
  see the README for its format.
- Fail before downloads when the checkpoint or identity metadata is missing.
  Do not substitute random parameters or overwrite the source weights.
- Restore weights, sample a training-only toy graph, and recompute its encoder
  outputs. Toy outputs need not equal full-graph outputs.
- Check entity and relation branches separately. Require finite outputs and
  finite, nonzero gradients into each branch's inputs and both projections.
  Also check the encoder/entity table for the entity branch and every selected
  scorer relation row for the relation branch.
- Require RoBERTa to remain frozen, in eval mode, with no parameter gradients.
  The squared-output loss is a gradient diagnostic; this gate does not perform
  optimizer steps, integrated triple scoring or benchmark evaluation.
- Save a unique success report containing config, code provenance, checkpoint
  and text hashes, remapping and gradient norms. A failed attempt must not
  produce a success report.

## Workflow and verification

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_phase3_gate.py" -v
python run_phase3_smoke_test.py
```

Offline tests use trained synthetic R-GCN fixtures and a tiny random RoBERTa,
not downloaded pretrained weights. Keep this distinction explicit in reports.
Use the CPU for offline checks and a Colab GPU for real-data/pretrained-LM
verification. Runtime changes in Colab wipe its local files and installed
packages; preserve artifacts in persistent storage.

Keep model, preprocessing, training and evaluation logic modular. Put run
settings in YAML and use thin entry-point scripts. Document shapes at module
boundaries. Avoid unnecessary abstractions and new dependency stacks.

## Remaining work and constraints

1. Recover the R-GCN/128 checkpoint and original ID maps, or run and record a
   new baseline that embeds those maps. Run the real Phase 3 gate and preserve
   its report before declaring that gate passed.
2. Phase 4: add and validate graph encoder 2 using projected KG-space inputs.
3. Phase 5: assemble full DistMult training, durable/resumable checkpoints and
   final test evaluation of the best-validation checkpoint.
4. Phase 6: compare KG-only, text-enhanced and proposed models, add ComplEx,
   and run warm-up/dimension ablations and multiple seeds under matched settings.

The current trainer saves only at the end and lacks optimizer/RNG resume state.
Do not describe it as resumable. Evaluation currently uses optimistic tie ranks
and the trainer evaluates validation only. Full-graph memory and LM activation
memory must be measured before scaling beyond the small gate.

The projections currently use Linear, LayerNorm, GELU and Dropout; there is no
recorded ablation proving this better than linear-only projections. DistMult's
head/tail symmetry is a known model limitation. Formal ontology reasoning,
full LM fine-tuning, ULTRA and human-subject evaluation are outside the current
scope. Preserve the historical loss/dataset corrections documented in the
experiment register; the old long README's numbered references are obsolete.

For each future run record the exact revision/dirty state, config, dataset and
text fingerprints, ID maps, seed, environment/device, metrics by split, selected
epoch, artifact path/hash and durable log location. Keep research claims grounded
in real runs rather than fixture tests.
