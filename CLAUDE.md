# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Research codebase for a BSc (Hons) Software Engineering intermediate research
project at the University of Kelaniya: **"Integrating Structural and Textual
Semantics for Ontology-Enriched Knowledge Graph Representation Learning."**

**Current status: Phase 2 complete; Phase 3 next.** Reconciled on 2026-10-05
(Asia/Colombo) against source revision `4b694a2`.

Read [README.md](README.md) for the reconstructed research overview and
[experiments/RUNS.md](experiments/RUNS.md) for run settings, metrics, revision
provenance and checkpoint recovery. The original long research README was not
found; historical numbered README references below refer to that unavailable
document, not to sections in the reconstructed README.

Phase 0, the Phase 1 KG-only models, and the standalone Phase 2 semantic bridge
are implemented. Phase 1 R-GCN full training and Phase 2 local/T4 gates are
reported passed in the retained notes. No historical checkpoint or original
run log has been recovered in this checkout; exact executed revisions are
unknown. Historical "saved"/"passed" statements below are reports, not claims
of present local artifact availability. Revised full RGAT training remains
unverified. Phase 3 integration is the next implementation milestone, with
**R-GCN at KG dimension 128 selected** in
[experiments/configs/phase3_integration.yaml](experiments/configs/phase3_integration.yaml).
Use two layers, 30 bases and graph dropout 0.2 to match the full baseline;
the semantic bridge maps 128 -> 768 -> frozen RoBERTa -> 768 -> 128.
RGAT remains a separate comparison track. The Phase 3 config is a contract for
the future runner, not evidence that integration or checkpoint recovery passed.

## Core idea (one paragraph)

The project reverses JAKET's LM-centered `LM → GNN → LM` pipeline into a
KG-centered `KG → GNN → LM → GNN → KG` pipeline. A relation-aware graph
encoder (KG Encoder Phase 1) learns structure-aware embeddings from training
triples only, these are projected into a frozen language model's hidden space
and injected via soft-prompt conditioning, the LM output is projected back
into KG space, a second graph encoder (KG Encoder Phase 2) refines it, and a
KGE scorer (DistMult first, ComplEx later) scores triples for link
prediction. The graph representation — not the language representation — is
the final optimization target.

## Non-negotiable implementation rules

These come directly from supervisor-identified mistakes in the preliminary
experiment (see README.md §15–17) or from the methodology chapter. Violating
them reproduces already-diagnosed bugs.

1. **Never initialize KG embedding parameters directly from raw LM output.**
   The preliminary experiment did this (BERT → ComplEx embedding tables) and
   it performed 86–92% worse than random init. The corrected design always
   routes LM information through projection layers + a KG warm-up stage —
   never a direct parameter replacement.
2. **Match the loss to the scorer family.**
   - DistMult / ComplEx (bilinear/semantic-matching models) → BCE / logistic
     loss, framed as binary classification.
   - Margin-based ranking loss is for translational-distance models (TransE,
     RotatE) only. Using margin loss with ComplEx was the root cause of the
     first diagnostic failure — do not repeat it.
3. **Build the message-passing graph from training triples only.** Validation
   and test triples must never be encoder edges. Filtered ranking (removing
   other known-true triples from candidate lists) is the only place
   validation/test triples' *labels* are used, and only at evaluation time.
4. **Freeze the language model in the first implementation.** Only train: KG
   Encoder Phase 1, KG Encoder Phase 2, KG→LM projection, LM→KG projection,
   and the KGE scorer. Full LM fine-tuning is out of scope for the current
   Colab-constrained stage.
5. **Keep a fixed learning rate for the duration of a single run.** Changes
   across runs must be explicit hyperparameter changes, not within-run
   switching, so comparisons stay attributable.
6. **FB15k-237 (Freebase-derived) is this project's primary dataset, by
   explicit supervisor direction.** This decision has a history worth
   knowing so it isn't re-litigated by accident:
   - The original preliminary diagnostic experiment (README.md §15–17) used
     **WN18RR**.
   - Mid-project, the supervisor's WN18RR-vs-WN18 guidance (specific to that
     diagnostic experiment, about inverse-relation shortcuts hiding subtle
     effects) was generalized to make **WN18** the project's primary dataset.
     Phase 0 was fully built and gate-verified against real WN18 on this
     basis (40,943 entities / 18 relations / 141,442 / 5,000 / 5,000
     train/valid/test — confirmed against published numbers).
   - That was then superseded: the supervisor specified **Freebase**, and
     the team's own reasoning converged on the same conclusion — this
     project's core question is whether *textual* semantics help KG
     embeddings, and WordNet-style entities (single words, one-line
     dictionary definitions) give a language model far less real signal to
     work with than Freebase-style entities (real-world things with
     Wikipedia-length descriptions). **FB15k-237** is the standard
     Freebase-derived KGC benchmark (not CoDEx-M, which is Wikidata-derived
     — a teammate's exploratory notebook used both, but "Freebase" points to
     FB15k-237 specifically) and is what most of the KG-LM literature this
     project's own review cites (KEPLER, ERNIE, KG-BERT, JAKET) evaluates on.
   - Phase 0 was rebuilt a second time for FB15k-237 (`preprocessing/dataset.py`,
     `experiments/configs/phase0_fb15k237.yaml`) and **verified in Colab**:
     stats matched FB15k-237's published numbers exactly (14,541 entities /
     237 relations / 272,115 / 17,535 / 20,466) on the first attempt — no
     repeat of WN18's column-order bug this time.
   - When interpreting or reporting results: note the dataset choice and its
     rationale explicitly wherever results are written up, don't silently
     swap datasets between comparable runs (every run in the validation
     table, README.md §12, should use the same dataset unless the
     comparison is specifically about dataset choice), and don't directly
     compare our FB15k-237 numbers against WN18/WN18RR numbers from the
     preliminary experiment — different datasets, not a fair comparison.
7. **Encode the graph once per optimizer step** and reuse representations for
   sampled positive/negative triples in that step — don't recompute the full
   graph encoding per triple.

## Architecture pipeline (implementation order)

```
KG triples + entity/relation descriptions
  → entity/relation ID mapping, train/val/test split, graph adjacency build
  → random KG embedding init (practical dim: 256)
  → [optional] KG warm-up: train KG side on structure only, no LM involved
  → KG Encoder Phase 1 (R-GCN / Relational GAT / CompGCN)
  → KG→LM projection (e.g. 256 → 768, trainable linear)
  → inject into frozen LM via soft-prompt injection
  → frozen LM forward pass → KG-aware semantic embedding
  → LM→KG projection (e.g. 768 → 256, trainable linear)
  → KG Encoder Phase 2 (refines the projected representation)
  → KGE scorer (DistMult first, ComplEx later)
  → BCE/logistic loss with filtered negative sampling
  → evaluate: filtered MRR, Hits@1, Hits@3, Hits@10
```

Recommended build order (don't implement out of sequence — each stage should
be validated before the next is added): KG-only baseline → confirm training
stability → add KG warm-up → Encoder Phase 1 → KG→LM projection → frozen LM +
soft-prompt injection → LM→KG projection → Encoder Phase 2 → DistMult
evaluation → ComplEx evaluation → ablations (no-warmup, embedding dim
128/200/256).

## Validation/ablation targets to keep in mind

When adding experiment configs, these are the comparisons the research plan
calls for (README.md §12): KG-only baseline, text-enhanced baseline (LM
description embeddings + KG model, no bookended architecture), full proposed
model with DistMult, full proposed model with ComplEx, ablation without KG
warm-up, ablation across embedding dimensions {128, 200, 256}.

## Reproducibility — record per run

Dataset + split version, random seed, KG embedding dim, LM model, KG encoder
type, attention heads, learning rate, optimizer, batch size, epochs, negative
sampling method, loss function, warm-up on/off, KGE scorer, evaluation
metrics, checkpoint path, final results. (README.md §25 has the full list —
put this in a config file per run, not just logs.)

## Environment constraints

Target environment is Google Colab (Tesla T4 in the preliminary experiment) —
single GPU, limited memory. Design choices should stay consistent with this:
smaller embedding dims over the JAKET-original 768, frozen LM, controlled
batch sizes, filtered/subsampled graphs rather than full Wikidata scale.

## Limitations (design around these, don't try to "fix" them by scope creep)

- **Compute ceiling**: single T4-class GPU in Colab, not JAKET's multi-GPU
  pretraining setup. No full-corpus pretraining — everything is filtered
  subgraphs + controlled dims.
- **Colab sessions are ephemeral**: runtime disconnects/resets lose GPU state
  and unsaved variables. Code must checkpoint to persistent storage (Drive)
  regularly, not just at the very end of a run.
- **Data quality**: entity-linking (SLING) noise and Wikipedia description
  sparsity for a portion of Wikidata entities reduce usable graph-text pairs.
  Don't assume every entity has a good description at inference/training time
  — handle missing/short descriptions explicitly (e.g. fallback text or
  entity-only path) rather than crashing.
- **ULTRA (pretrained transferable GNN) is not integrated.** It's noted as a
  future direction only; the current encoders train from scratch per target
  KG. Don't add ULTRA-specific code paths unless explicitly asked.
- **No formal ontology / description-logic support.** Scope is standard
  relational KGs (triples), not OWL/EL++ style ontologies.
- **No human-subject evaluation at this stage** — evaluation is purely
  link-prediction metrics (MRR, Hits@K) on held-out triples, not user studies.
- **The preliminary WN18RR/BERT+WordNet diagnostic numbers are not a valid
  baseline to compare against.** They were produced with a mismatched loss
  function and untuned hyperparameters (see Non-negotiable rule #2). Never
  cite them as "the LM-init result" — they were explicitly corrected/discarded
  as methodology, only kept as a documented diagnostic.

## Tools & Techniques

**Core stack:** Python, PyTorch, Hugging Face Transformers (frozen LM
loading), PyTorch Geometric or DGL (relation-aware graph encoders), TorchKGE
(used for the ComplEx prototype — fine to reuse for KGE baselines/scoring),
NumPy/Pandas (data prep), Matplotlib (loss curves, metric comparison plots),
Google Colab (execution environment), GitHub (version control).

**Modeling techniques in play, by pipeline stage:**
- Graph encoders (KG Encoder Phase 1 & 2): relation-aware message passing —
  R-GCN, Relational GAT, or CompGCN. Pick one to start (R-GCN is the simplest
  to debug) rather than building all three at once.
- Embedding-space bridging: trainable linear projection layers (KG dim ↔ LM
  hidden dim), not learned nonlinear adapters unless a linear projection
  demonstrably underfits.
- LM conditioning: soft-prompt injection (projected KG vector prepended as a
  pseudo-token before the description text), LM weights frozen
  (`requires_grad_(False)`, `model.eval()` during forward passes used for
  conditioning).
- Scoring: DistMult first (bilinear, cheap, easy to debug), ComplEx later
  (complex-valued, models asymmetric relations).
- Training objective: BCE/logistic loss over positive + corrupted-negative
  triples, Bernoulli or filtered negative sampling.
- Evaluation: filtered MRR, Hits@1/3/10 — filtering removes other known-true
  triples from the candidate ranking, computed only at eval time.

## Implementation Strategy — Phased Build & Test (do not build the whole pipeline at once)

Build one stage at a time, prove it works in isolation, checkpoint it, then
integrate. This mirrors exactly how the preliminary experiment's failure was
diagnosed — by isolating which stage (loss function, in that case) was wrong.
Don't write Phase 1 + LM + Phase 2 + scorer all before running anything.

**Gate discipline:** each phase below ends with a smoke test that must pass
before starting the next phase. A "smoke test" here means: a tiny synthetic
or heavily subsampled graph (dozens of entities, not the full dataset), run
for a handful of steps, checked for correct tensor shapes, finite/non-NaN
loss, loss trending down, and no accidental val/test leakage — not a full
training run. Full-dataset training only happens after the smoke test passes.

1. **Phase 0 — Data & config scaffolding. ✅ Done — gate passed on real
   FB15k-237 data.**
   Entity/relation ID mapping, train/val/test split, adjacency structures
   built from *training triples only*, a config file holding the run
   settings, and a small connected toy subset for smoke testing later
   phases. See "Repository Map" below for the exact files.
   *Gate:* `run_phase0_smoke_test.py` run in Colab on real FB15k-237 and
   printed PASSED, with dataset statistics matching the published numbers
   exactly on the first attempt (14,541 entities / 237 relations /
   272,115 / 17,535 / 20,466 train/valid/test) — no repeat of the WN18
   column-order bug (see rule #6 for that history). Toy subset also
   verified self-consistent (50 entities, own fresh id range, non-empty
   train/valid/test).

2. **Phase 1 — KG-only baseline (Encoder Phase 1 + scorer, no LM at all).
   ✅ Done — both toy-subset and full-scale gates PASSED in Colab.**
   Relation-aware R-GCN encoder + DistMult scorer, built as a standalone
   trainable model (`models/kg_encoder.py`, `models/scorer.py`,
   `models/kg_only_baseline.py`), trained via a config-driven loop
   (`training/train_kg_baseline.py`) with BCE loss (`training/losses.py`)
   and filtered negative sampling (`training/negative_sampling.py`), scored
   with filtered MRR/Hits@K (`evaluation/metrics.py`). See "Repository Map"
   below for the full file list.
   *Toy gate:* `run_phase1_smoke_test.py` run in Colab against
   `experiments/configs/phase1_toy.yaml` (dim=32, 20 epochs) and printed
   PASSED — loss finite throughout, loss improved (early avg 0.6944 → late
   avg 0.0033; the near-zero end loss is expected memorization of a 72-triple
   toy set, not a meaningful result), validation metrics computed without
   error, checkpoint reload verified identical. Note validation MRR did
   **not** improve monotonically across evaluations (0.2546 → 0.3174 →
   0.3214 → 0.3169) — this directly motivated adding best-checkpoint
   tracking (see below) before the real run, rather than just saving
   whatever the last epoch happens to produce.
   *Real run:* `experiments/configs/phase1_full.yaml` (real FB15k-237, no
   toy subset, dim=128 — see below for why this was lowered from the
   originally-planned 256 — 100 epochs, `save_best: true`) plus
   `run_phase1_full_training.py` to launch it — use this runner script
   rather than `python training/train_kg_baseline.py ...` directly, since
   the latter breaks the package-relative imports (Python puts the script's
   own folder, not the repo root, on `sys.path` in that case). Needs a GPU
   Colab runtime, unlike everything before this point. Produces
   `experiments/checkpoints/kg_only_baseline.pt` — the real "KG-only
   baseline" row in the validation table, not just a stepping stone.
   *Gate (for the real run):* stable training curve, reasonable filtered
   MRR/Hits@K, best-validation checkpoint saved and reloadable.
   *First attempt hit a CUDA OOM* inside `RGCNConv.forward` (`h @
   weight[i]`), on a T4-class GPU (~13GB used at OOM). Root cause: PyG's
   `RGCNConv`, for each of the 474 message-passing relations (237 × 2 for
   inverse edges) in turn, runs a full graph-propagate step that produces
   a dense `[num_entities, dim]` tensor, then multiplies it by that
   relation's weight — and autograd has to keep **all 474** of those
   per-relation `[14541, dim]` tensors alive simultaneously for backward.
   That memory scales with `dim`, not with parameter count.
   *First fix attempt (num_bases) was insufficient* — added `num_bases: 30`
   (basis decomposition) expecting it to fix this, but it doesn't: basis
   decomposition only shrinks the *weight* tensors (474 separate `[dim,
   dim]` matrices → 30 shared ones), not the 474 per-relation activation
   tensors described above, which are the actual bulk of the memory. Confirmed
   by a second Colab run that hit the identical OOM (same ~13GB, same free-memory
   figures) even with `num_bases` set — proof the fix had no real effect.
   *Actual fix:* dropped `model.dim` from 256 to 128 in `phase1_full.yaml`
   (halves the size of every one of those 474 per-relation tensors); kept
   `num_bases: 30` too since it's still a real, if secondary, memory saving
   on the weight tensors themselves. Drop to 64 if 128 still OOMs. Batch
   size was never the cause — the graph is encoded once per optimizer step
   (rule #7) regardless of batch size, so this OOM was independent of it.
   *Second issue, after the OOM fix: ~1hr for 4 epochs on a T4* — not a
   hardware problem, a batch-size/architecture mismatch. The encoder
   re-runs a full graph pass (all 14,541 entities, ~544k message-passing
   edges, looping internally over all 474 relation types) **once per
   batch**, correctly per rule #7 — but at `batch_size: 512` that's
   272115/512 ≈ 532 of those expensive full-graph passes every single
   epoch, each ~950 relation-loop operations (474 relations × 2 layers),
   which is what actually eats the time — scoring a batch's triples with
   the resulting embeddings is comparatively cheap. Fixed by raising
   `batch_size` to 32768 (~9 full-graph passes/epoch instead of 532, ~59x
   fewer) — go even higher, or to `batch_size >= len(train)` for one
   encode per epoch (standard for full-batch R-GCN training in the
   original R-GCN paper), if still slow. If epoch time is *still* high
   after this, the next suspect is `training/negative_sampling.py`'s
   pure-Python per-triple rejection-sampling loop (unrelated to batch
   size — same total work either way).
   *Result: the `batch_size: 32768` fix worked and the run completed in
   Colab.* Full 100-epoch training curve was stable throughout — loss fell
   monotonically (1.3855 → 0.0948), no NaNs/spikes. Validation MRR rose
   0.0511 (epoch 5) → 0.1842 (epoch 100) and Hits@10 rose 0.1096 → 0.3409,
   with the best-checkpoint tracking correctly identifying epoch 100 itself
   as the best (val_MRR was still climbing at the end, not yet plateaued —
   loss was too, so more epochs would likely improve this further, noted
   here rather than acted on since it's a hyperparameter-investment
   decision, not a gate-blocking bug). Checkpoint saved to
   `experiments/checkpoints/kg_only_baseline.pt`. This is somewhat below
   fully-tuned published DistMult/R-GCN numbers on FB15k-237 (papers:
   roughly 0.24–0.31 MRR) — expected given `dim=128` (halved from the
   originally-intended 256 for GPU memory, see above) and only
   `num_negatives: 4` (papers typically use far more), not a sign of a
   bug. Gate considered passed: training was stable, metrics were far
   above random-guessing level and moved consistently in the right
   direction, and the checkpoint saved successfully.
   *RGAT variant (added alongside, not a replacement):* by explicit user
   decision, a second Phase 1 encoder — `models/kg_encoder_rgat.py`
   (`RGATEncoder`, using PyTorch Geometric's `RGATConv`) and
   `models/kg_only_baseline_rgat.py` (`KGOnlyBaselineRGAT`) — was built
   alongside R-GCN rather than replacing it, so both exist as separate
   checkpoints/configs for a later encoder comparison. Selected via
   `model.encoder_type: rgat` in a config, resolved by the new
   `training/model_factory.py` (shared by `train_kg_baseline.py` and both
   smoke-test scripts so they can't disagree on how a config becomes a
   model). R-GCN stays the primary/default baseline — nothing about its
   config or checkpoint changed. Has its own toy config
   (`experiments/configs/phase1_rgat_toy.yaml`) and gate script
   (`run_phase1_rgat_smoke_test.py`), mirroring the R-GCN toy gate exactly
   (same phased-build discipline: toy smoke test before any real run).
   RGAT is *at least* as likely as
   R-GCN was to hit GPU memory/speed issues at full scale (attention adds
   more per-relation state on top of R-GCN's already-expensive per-relation
   loop) — expect to revisit `dim`/`batch_size`/`num_bases` again for it
   specifically, don't assume the R-GCN full-run config's values transfer.
   *RGAT toy gate, first attempt: training passed, checkpoint-reload check
   FAILED* — loss finite/improving and validation metrics computed
   correctly, but `encoder.layers.0.l2` and `encoder.layers.1.l2` (shape
   `(dim, dim)`) came back with `max abs diff=nan` between the trained
   model and the reloaded one. This surfaced a real latent bug in the
   smoke test's comparison itself: it compared parameters by **position**
   (`zip(a.values(), b.values())`) instead of by name, so a mismatch could
   never be more specific than a single opaque pass/fail — fixed in
   `run_phase1_smoke_test.py` to compare by key and report the exact
   mismatched name(s) and max diff (this fix is unconditionally better and
   applies to the R-GCN gate too, not just RGAT's). That's what surfaced
   `l2` specifically instead of a bare failure. Root cause (confirmed by
   reading PyTorch Geometric's actual `RGATConv` source, not guessed):
   `RGATConv` always allocates four parameters — `l1`, `b1`, `l2`, `b2` —
   for an optional `mod="scaled"` attention variant, even when that mode
   isn't used. We never pass `mod`, so these are dead weight in our
   forward pass regardless. They're supposed to get a harmless constant
   init (`l2` filled with `1/out_channels`) either way, but that
   apparently isn't landing cleanly as finite on whatever PyTorch Geometric
   version Colab installed. *Fix:* `models/kg_encoder_rgat.py` now zeroes
   out and freezes (`requires_grad_(False)`) these four dead parameters
   itself right after constructing each `RGATConv` layer, rather than
   depending on upstream's initialization for a code path this project
   never exercises — guarantees every parameter is finite and reproducible
   regardless of installed PyG version.
   *RGAT toy gate: ✅ PASSED in Colab after the fix.* Loss finite throughout
   and improved (early avg 1.1134 → late avg 0.1142 — a 50-entity toy set
   like R-GCN's, so this is expected memorization, not a meaningful
   result), validation metrics computed without error, checkpoint reload
   now verified identical (the `l2` mismatch is gone — confirms the fix
   worked, not just moved the problem). Checkpoint saved to
   `experiments/checkpoints/phase1_rgat_toy.pt`. RGAT is cleared for a real
   full-dataset run.
   *RGAT real run, first attempt (dim=128, heads=2, copied from R-GCN's
   tuned config) hit a CUDA OOM* trying to allocate **66.44 GiB** in
   `RGATConv.message()`'s `torch.index_select(w, 0, edge_type)` — that line
   builds one dense tensor holding a separate weight matrix for *every
   message-passing edge at once*, shape `[num_edges, dim, heads*dim]`. With
   ~544k edges (272,115 triples × 2 for inverse) this is
   `num_edges * heads * dim^2 * 4 bytes`, a fundamentally worse scaling
   than R-GCN's OOM (which scaled with `relations * nodes`, not
   `edges * dim^2`, and was fixable with a modest `dim` cut). `num_bases`
   does **not** help here — it only shrinks the per-relation weight table
   *before* this per-edge expansion happens, not the expansion itself.
   *Fix:* `phase1_rgat_full.yaml` now uses `dim: 32` (not 64 or 128) and
   `heads: 1`, both of which scale this specific tensor down directly.
   **Honest caveat, not swept under the rug:** this leaves RGAT's real run
   at meaningfully lower capacity than R-GCN's `dim=128` baseline — any
   performance gap between them could partly reflect "less capacity" rather
   than purely "different architecture," and this should be stated
   explicitly if/when comparing their results. Unlike R-GCN, there wasn't a
   "keep full capacity, tune something else instead" option available here
   — the only other real fix would be implementing edge/neighbor
   mini-batching inside `RGATEncoder` itself (an actual code change, not a
   config one), which hasn't been attempted. Not yet re-run with the
   dim=32/heads=1 fix.

3. **Phase 2 — KG ↔ LM semantic bridge (KG → LM projection + frozen
   RoBERTa + LM → KG projection, developed and tested independently before
   connecting real Phase 1 embeddings).
   ✅ Done — the standalone semantic bridge for both entity and relation text
   passed locally and on a Colab T4 GPU.**

   Phase 2 was started only after the structural KG side had been established
   in Phase 1. The purpose of this phase was not yet to train the complete
   KG → LM → KG architecture or to evaluate link-prediction performance.
   Instead, Phase 2 focused specifically on building and validating the middle
   semantic bridge of the architecture. The main question at this stage was:
   can a structural KG representation be taken from the 32-dimensional KG
   embedding space, converted into the 768-dimensional representation space
   used by a pretrained Language Model, combined with real textual information,
   processed by a frozen Language Model, and then converted back into the
   original 32-dimensional KG space without breaking the gradient path that
   will later be required for end-to-end training? To answer this question
   clearly, Phase 2 was developed independently from the real Phase 1 encoder.
   Dummy structural vectors with the correct dimensionality were used during
   the Phase 2 smoke tests so that any error could be identified as a Phase 2
   problem instead of being mixed with Phase 1 encoder behaviour.

   The first component implemented for Phase 2 was the KG-to-LM projection in
   `models/kg_lm_projection.py`. The structural representations produced by
   the KG side and the hidden representations expected by the Language Model
   do not have the same dimensionality. The revised Phase 1 RGAT configuration
   specifies KG vectors of dimension 32, while `roberta-base`
   operates with hidden representations of dimension 768. Therefore, a raw
   32-dimensional KG vector cannot be supplied directly as a RoBERTa embedding.
   To solve this, `KGLMProjection` was implemented as a trainable projection
   module that maps each structural representation from 32 dimensions to
   768 dimensions. The transformation uses a linear projection followed by
   normalization, GELU activation, and dropout. The important idea is that
   this projection is not a manually defined conversion. Its weights are
   trainable so that, once the complete model is integrated, the
   link-prediction objective can teach the projection how structural graph
   information should be represented inside the Language Model's embedding
   space.

   The projected 768-dimensional KG vector is then treated as a **soft prompt**.
   We do not convert the KG embedding into a sentence, and we do not try to
   assign the vector to an existing RoBERTa vocabulary token. Instead, the
   projected KG representation itself is inserted directly as an additional
   continuous embedding at the beginning of the Language Model input sequence.
   This gives the Language Model direct access to the structural information
   learned on the KG side. At the same time, the textual description of the
   corresponding entity or relation is tokenized normally using the RoBERTa
   tokenizer. RoBERTa's normal token embeddings are obtained for those text
   tokens, and the KG-derived soft prompt is prepended before them. Conceptually,
   the transformer therefore receives a sequence containing the KG structural
   prompt first and the textual description tokens after it. This is the main
   point where the structural KG representation and textual semantics are
   allowed to interact.

   The Language Model wrapper was implemented in `models/frozen_lm.py`. The
   first prototype of the bridge was used to verify that the general mechanism
   of inserting a projected KG representation into a pretrained transformer
   was workable. The final Phase 2 implementation was then standardized on
   `roberta-base`, whose hidden size is 768. RoBERTa is used as a pretrained
   semantic transformation module rather than a model that we fine-tune.
   Consequently, all RoBERTa parameters are frozen by setting their
   `requires_grad` values to `False`. This means that the pretrained RoBERTa
   weights remain unchanged when the complete KG model is later trained.
   However, an important technical requirement discovered and verified during
   Phase 2 is that freezing RoBERTa is not the same as wrapping the complete
   RoBERTa forward pass inside `torch.no_grad()`. If `torch.no_grad()` were
   used around the transformer computation, the computational graph would be
   broken and the final task loss would not be able to propagate gradients
   backward through the Language Model operations to the trainable KG-to-LM
   projection or to the upstream KG components. Therefore, the implementation
   freezes only the RoBERTa parameters while allowing autograd to track the
   operations performed on the trainable input embeddings. In this design,
   gradients are allowed to pass through RoBERTa, but RoBERTa's own parameters
   do not receive trainable gradients and are never updated.

   RoBERTa is also intentionally kept in evaluation mode even when the
   surrounding bridge is placed into training mode. This was done because the
   Language Model is supposed to behave as a fixed pretrained semantic module.
   Keeping it in evaluation mode disables training-time behaviour such as
   dropout inside RoBERTa and gives a stable transformation for the same input.
   The surrounding projection layers remain trainable, while the internal
   Language Model remains frozen and deterministic. This behaviour was not
   assumed; it was explicitly checked during the Phase 2 smoke tests.

   Because a custom KG soft prompt is inserted before the normal text-token
   embeddings, the attention mask produced by the tokenizer also has to be
   modified. The original mask contains positions only for the textual tokens,
   while the actual embedding sequence now contains one additional position
   for the KG prompt. Phase 2 therefore extends the attention mask by one
   active position so that the prompt participates correctly in the
   transformer computation. The combined embeddings are supplied to RoBERTa
   through `inputs_embeds`, allowing the manually created KG soft prompt to
   appear in the same sequence as the standard RoBERTa token embeddings.

   After this combined sequence passes through RoBERTa, the transformer returns
   contextual hidden representations for all positions in the sequence. Phase 2
   uses the contextual representation corresponding to the inserted soft-prompt
   position as the Language Model output for the KG item. Before entering
   RoBERTa, that position contains only the projected structural representation.
   After passing through the transformer, its representation has been
   contextualized through attention to the entity or relation description.
   This gives a 768-dimensional semantic representation that has been influenced
   by both the original graph structure and the textual context.

   The next component implemented was the LM-to-KG projection in
   `models/lm_kg_projection.py`. The contextual representation returned by
   RoBERTa still has dimension 768, but the KG architecture operates in a
   32-dimensional embedding space. `LMKGProjection` therefore performs the
   reverse mapping from 768 dimensions back to 32 dimensions. Like the first
   projection, this is a trainable transformation using a linear layer,
   normalization, GELU activation, and dropout. The output of this module is
   therefore once again compatible with the KG side of the architecture.
   This is important to the research design because the Language Model is not
   intended to become the final representation space. The architecture starts
   in KG space, temporarily moves into LM space to introduce textual semantics,
   and then deliberately returns to KG space so that graph refinement and
   link-prediction scoring can continue afterwards.

   The KG-to-LM projection, frozen Language Model, and LM-to-KG projection were
   then combined into the reusable `KGLMBridge` implemented in
   `models/kg_lm_bridge.py`. This wrapper provides the complete Phase 2
   transformation. A 32-dimensional structural vector and the associated text
   are supplied to the bridge, the structural vector is projected to 768
   dimensions and inserted as a soft prompt, RoBERTa processes the soft prompt
   together with the textual tokens, the prompt-position contextual
   representation is extracted, and that 768-dimensional representation is
   projected back to 32 dimensions. Therefore the overall dimensional flow of
   Phase 2 is `32 → 768 → RoBERTa → 768 → 32`. The bridge was deliberately
   written generically rather than as an entity-only implementation because
   the same mechanism is required for relations as well.

   Before real FB15k-237 descriptions were introduced, a generic Phase 2 smoke
   test was created to verify that the bridge itself worked. The smoke test
   supplied dummy 32-dimensional structural representations and passed them
   through the complete semantic bridge. It checked that the input and output
   shapes were correct, that no NaN or infinite values were produced, that
   RoBERTa remained frozen and in evaluation mode, that the trainable
   projection modules received gradients, and that a gradient could also
   propagate back to the original structural input. At the same time, the test
   verified that none of RoBERTa's own parameters received gradients. A small
   artificial loss such as `output.pow(2).mean()` was used only so that
   `.backward()` could be called and the gradient path could be inspected.
   This loss is not the research training objective, is not BCE, and has
   nothing to do with link-prediction quality. Its only purpose was to prove
   that the bridge remained differentiable in exactly the places where later
   end-to-end training will require differentiation.

   During the development of Phase 1 RGAT, GPU-memory behaviour forced the
   full-scale RGAT configuration to use `dim: 32` and `heads: 1`. Phase 2's
   smoke tests were aligned with this intended dimension. Successful full-scale
   RGAT training with that config has not been recorded, so feasibility remains
   unverified. The tested bridge path accepts `[batch, 32]`, creates
   `[batch, 768]` soft prompts, and returns `[batch, 32]`. `KGLMBridge` accepts
   a configurable `kg_dim`; these test settings do not fix the architecture to
   32. The later Phase 3 selection is R-GCN at dimension 128, matching the
   reported full baseline; this does not change the historical Phase 2 tests.

   After the generic bridge behaviour was established, the next major task was
   to replace placeholder text with real entity textual information. Entity
   text preprocessing was implemented in `preprocessing/entity_text.py`. The
   textual information is not generated by our model. Instead, existing
   external text resources corresponding to the same Freebase entities used by
   FB15k-237 are loaded and aligned with the dataset. Two forms of entity text
   are supported. Richer long descriptions are preferred when available, and
   shorter entity text is used as a fallback for entities that do not have a
   long description. The short text comes from the KG-BERT-style
   `entity2text.txt` resource, while longer Freebase descriptions are obtained
   from the `FB15k_mid2description.txt` resource. The purpose of using the
   long-description-first strategy is to provide RoBERTa with richer semantic
   context wherever possible while still guaranteeing that no entity is lost
   because a long description happens to be unavailable.

   A critical part of this preprocessing was preserving the original
   FB15k-237 entity mapping. Phase 2 does not create a new entity-ID order from
   the text files. The `entity2id` mapping already created by the KG data
   pipeline remains the single source of truth. Every description is mapped
   back to that existing ID so that the structural vector at entity index `i`
   and the textual description at entity index `i` always refer to exactly the
   same entity. This alignment step is important because an incorrectly aligned
   dataset could still produce tensors with valid dimensions and therefore
   pass simple shape tests while silently combining one entity's structural
   embedding with another entity's description. The text preprocessing code
   also cleans formatting artefacts such as `@en` markers, quotation
   formatting, literal or real newline characters, and unnecessary whitespace
   before the descriptions are passed to the tokenizer.

   The entity-text alignment was checked explicitly rather than assuming that
   all FB15k-237 entities had descriptions. The final result was full coverage
   across all 14,541 FB15k-237 entities. Long descriptions were found for
   14,515 entities and the remaining 26 entities were successfully covered
   using the short-text fallback. As a result, there were zero missing entity
   descriptions and the final alignment coverage was 14,541 out of 14,541,
   or 100%. The associated validation scripts include the entity-text
   alignment check, long-text alignment check, and the real-text Phase 2 smoke
   test.

   Once entity-text alignment had been verified, a second level of Phase 2
   testing was performed using real FB15k-237 descriptions. The structural
   side was still intentionally represented by dummy 32-dimensional vectors
   because real Phase 1 integration was not part of this phase yet. The test
   therefore exercised the actual semantic path: a correctly shaped KG vector
   was projected into RoBERTa space, a real aligned entity description was
   tokenized, the structural vector was inserted as a soft prompt before the
   token embeddings, the combined sequence was processed by frozen RoBERTa,
   and the contextual prompt representation was projected back to a
   32-dimensional KG vector. The test again confirmed valid shapes, finite
   outputs, correct gradient flow through both projections, and the absence of
   gradients on RoBERTa parameters. The entity-side bridge passed in the local
   environment and was then independently run successfully on the Colab T4 GPU
   environment.

   After the entity path was working, relation semantics were added because the
   final KGE model cannot rely only on semantically enriched entities. Relations
   also participate directly in triple scoring, so the architecture requires a
   textual-semantic path for relation representations as well. Relation text
   preprocessing was therefore implemented in
   `preprocessing/relation_text.py`. As with entities, the existing
   `relation2id` mapping is preserved and no new independent relation ordering
   is introduced. Textual forms for the 237 FB15k-237 relations are loaded from
   the corresponding relation-text resource. The preprocessing can also convert
   a Freebase-style relation identifier into a readable phrase when necessary;
   for example, a relation path such as
   `/location/country/form_of_government` can be converted to text similar to
   `location country form of government`. This conversion changes only the
   textual representation supplied to the Language Model and does not change
   the original relation identity used by the KG.

   The relation-text alignment was also explicitly validated. All 237 original
   FB15k-237 relations were successfully matched to textual representations,
   leaving zero missing relation descriptions and giving 237 out of 237, or
   100%, relation-text coverage. A dedicated relation-text smoke test was then
   run using real relation text together with dummy 32-dimensional structural
   relation vectors. These vectors passed through the same `KGLMBridge` used
   for entities. This verified that the bridge was not accidentally dependent
   on entity-specific assumptions and could process relations through the same
   `32 → 768 → RoBERTa → 768 → 32` path. The relation-side test verified valid
   tokenization, correct structural and output dimensions, finite outputs,
   gradients for the trainable projection modules, and no gradients for the
   frozen RoBERTa parameters. This relation-side gate also passed both locally
   and on the Colab T4 GPU.

   One important integration detail identified during Phase 2 concerns where
   the real structural relation vectors will come from. The Phase 1 RGAT
   encoder directly produces structural entity representations, but it does
   not independently output a simple `[237, 32]` final relation representation
   matrix in the same way. The KGE scorer, however, contains trainable relation
   embeddings. These learned relation embeddings are therefore the intended
   structural relation input to the semantic bridge once the phases are
   integrated. During Phase 2 this real connection was deliberately not made;
   dummy 32-dimensional relation vectors were sufficient to validate that the
   semantic bridge itself works for relation inputs. Connecting the actual
   learned relation embeddings to their aligned descriptions belongs to the
   integration phase that follows Phase 2.

   By the end of Phase 2, the complete standalone semantic mechanism had
   therefore been implemented and verified. For entities, a 32-dimensional
   structural vector can be projected into RoBERTa's 768-dimensional space,
   combined with a correctly aligned real entity description, processed
   through the frozen transformer, and returned as an enriched
   32-dimensional KG representation. The same mechanism has also been verified
   for relations using real relation text. Entity-text alignment reached
   14,541 out of 14,541 entities, relation-text alignment reached 237 out of
   237 relations, both sides have 100% textual coverage, the trainable
   projections receive gradients, RoBERTa remains frozen, and the complete
   standalone path works both locally and on the target Colab T4 environment.

   The forward flow established by Phase 2 can therefore be summarized as:

   ```text
   structural KG vector [32]
            ↓
   KG → LM projection
          32 → 768
            ↓
   KG-derived soft prompt [768]
            +
   aligned entity/relation description
            ↓
   RoBERTa token embeddings
            ↓
   soft prompt prepended to text sequence
            ↓
   frozen roberta-base
            ↓
   contextual prompt representation [768]
            ↓
   LM → KG projection
          768 → 32
            ↓
   semantically enriched KG representation [32]
   ```

4. **Phase 3 — Integrate Phase 1 → Phase 2 (KG → LM).**
   Use the selected R-GCN/128 contract in `phase3_integration.yaml`, matching
   `phase1_full.yaml`. Recover and validate `kg_only_baseline.pt` first; it is
   not present locally. Feed real entity outputs and scorer relation embeddings
   from that model into the
   now-validated LM module. Run a small forward/backward smoke test and
   confirm gradients flow into Encoder Phase 1 and the projection layers but
   not into the frozen LM.
   *Gate:* integrated forward/backward runs cleanly on the toy subset with no
   shape or gradient-flow surprises.
   
5. **Phase 4 — Add KG Encoder Phase 2 (LM → KG refinement).**
   Build Encoder Phase 2 standalone first (dummy projected KG-space vectors),
   then wire it onto the real output from Phase 3.
   *Gate:* standalone smoke test passes, then integrated smoke test passes.

6. **Phase 5 — Full pipeline + scorer + training loop.**
   Assemble KG Encoder 1 → LM → KG Encoder 2 → DistMult end-to-end. Smoke
   test on the toy subset, then run on the real dataset. Compare against the
   Phase 1 checkpoint (KG-only baseline) using the same eval protocol.
   *Gate:* full pipeline trains stably and produces filtered MRR/Hits@K
   comparable in form to the baseline run (better or worse is a research
   result, not a bug — but NaNs, collapsed embeddings, or wildly
   out-of-range metrics mean something's broken).

7. **Phase 6 — Ablations & ComplEx.**
   Only after Phase 5 is stable: swap in ComplEx, run the no-warm-up
   ablation, run the embedding-dimension ablation ({128, 200, 256}).

Optional KG warm-up (README.md §5) slots into Phase 1 as an initial
structure-only training stage before Encoder Phase 1's output is considered
"final" for that phase — implement and gate it as part of Phase 1, not as a
separate phase.

## Code Organization & Colab Guidelines

- **Modular, not monolithic.** Each concern gets its own file (encoder,
  projection, LM wrapper, scorer, losses, data prep, training loop, eval) —
  never accumulate everything in one script or one notebook cell. Follow the
  `models/ preprocessing/ training/ evaluation/` layout already sketched in
  README.md §23.
- **Notebooks stay thin.** A notebook (or Colab cell sequence) per phase
  (e.g. `notebooks/phase1_kg_baseline.ipynb`) should only import from the
  modular package and orchestrate calls (`!git clone` the repo or mount
  Drive + `sys.path.append`, then `from models.kg_encoder import ...`) — no
  business logic pasted inline into cells. This keeps it both readable and
  actually re-runnable outside Colab.
- **Config-driven, not hardcoded.** Hyperparameters live in a config
  file loaded once at the top of a run, never scattered as magic numbers
  across modules.
- **Shape/contract discipline at module boundaries.** Because the one
  documented near-miss in this project was a silent entity-ID/embedding
  misalignment, every module boundary (projection in/out, encoder in/out,
  LM wrapper in/out) should state expected tensor shapes in
  docstrings/type hints and assert them at runtime in debug/smoke-test mode.
- **Checkpoint after every phase gate**, to Drive-backed paths, not just
  local Colab disk — a disconnected runtime otherwise loses the work.
- **Keep dependencies Colab-installable** (`pip install` at the top of the
  notebook) and avoid anything requiring local-only setup (e.g. GPU-specific
  builds Colab doesn't ship) unless there's no alternative.

## Running This Project in Google Colab

This is the standard sequence for running the existing Phase 0, Phase 1 and
Phase 2 entry-point scripts in a fresh Colab runtime. See README.md for the
current gate commands and experiments/RUNS.md for expected artifacts.

**0. Pick the runtime type first** (Runtime → Change runtime type →
Hardware accelerator): Phase 0 and Phase 1's *toy-subset* smoke test need
**no GPU/TPU at all** (tiny data, tiny model) — select **None (CPU)** and
save your GPU quota. Switch to a GPU (T4 is what the report used) once
training on the *real, full* dataset (`run_phase1_full_training.py` onward),
and keep it on for Phase 2 onward once the frozen language model is
involved.

**Changing the runtime type restarts the VM and wipes everything** that
isn't in Google Drive — the cloned repo folder, every `pip install`ed
package, all in-memory variables. So switching hardware accelerator
mid-session (e.g. CPU → GPU right before the real Phase 1 run) means
redoing steps 1–2 below from scratch, not just re-running step 3. If this
back-and-forth gets annoying, mounting Google Drive and cloning the repo
there instead of Colab's local disk makes the clone (though not the pip
installs) survive a runtime restart.

**1. Clone the repo:**
```
!git clone https://github.com/Rashmika119/Research-project.git
%cd Research-project
```
Use `%cd` (a Colab "magic" command), not `!cd` — `!cd` only changes directory
inside that one throwaway subprocess and won't carry over to the next cell;
`%cd` actually changes Colab's working directory going forward.

If the repo is **private**, a plain `git clone` fails — Colab has no way to
prompt for a GitHub login interactively. Either:
- make the repo public on GitHub (Settings → Danger Zone → Change
  visibility) if nothing sensitive is in it, or
- clone with a Personal Access Token embedded in the URL:
  `!git clone https://<github-username>:<token>@github.com/Rashmika119/Research-project.git`
  (generate one under GitHub Settings → Developer settings → Fine-grained
  tokens; don't leave it sitting in a notebook cell you share with others).

**Running a feature branch instead of `main`:** everything above defaults to
`main`. To run a different branch (e.g. a teammate's in-progress work, or
your own experiment branch):
- Create + push it once, from wherever you edit (not Colab):
  ```
  git checkout -b feature/my-change
  # ... commit your changes ...
  git push -u origin feature/my-change
  ```
- Then in Colab, either clone it directly —
  `!git clone -b feature/my-change https://github.com/Rashmika119/Research-project.git` —
  or, if you already cloned `main` in that session, switch without
  re-cloning: `!git fetch origin` then `!git checkout feature/my-change`
  (run from inside the `Research-project` folder).
- Switching branches this way does **not** wipe pip installs or GPU state
  the way a runtime-type change does (see step 0 above) — it only changes
  which files are on disk. Only reinstall `requirements.txt` if the branch
  actually changed dependencies.

**2. Install dependencies.**
Only install what the phase you're running actually needs — don't wait on
the full stack if you don't have to:
- Phase 0 only needs `pyyaml` (`dataset.py`/`graph_builder.py`/`toy_subset.py`
  are otherwise plain Python): `!pip install pyyaml`
- For Phase 1 onward, install everything: `!pip install -r requirements.txt`
  (this takes longer, and `torch-geometric` occasionally needs a
  version-matched install command — see the comment in `requirements.txt`).

**3. Run the phase's entry-point script**, e.g.:
```
!python run_phase0_smoke_test.py
```
It prints a step-by-step summary ending in either a `PASSED` message or a
specific failure — paste whatever it prints back into the chat with Claude
if something goes wrong, rather than trying to debug it blind.

**If you hit a GitHub authentication error** unrelated to the above (e.g. a
`403`/permission error on push, not clone), that's almost always a cached
Git Credential Manager login for the wrong account on that machine — see the
git-remote/auth troubleshooting in this project's chat history, or ask
Claude to check `git remote -v` and `git config --global --list` first.

## Repository Map (what exists now, and why)

The map covers the implemented Phase 0–2 components. Runtime data, checkpoints
and logs are not present in this checkout; paths below describe their intended
locations. Historical verification claims are detailed in experiments/RUNS.md.

| File | Purpose |
|---|---|
| `requirements.txt` | Python packages needed across all phases (Colab-installable). |
| `.gitignore` | Keeps downloaded data, checkpoints, and caches out of version control. |
| `preprocessing/__init__.py` | Makes `preprocessing/` an importable package; just a module docstring. |
| `preprocessing/dataset.py` | Reads FB15k-237's raw text triples (primary dataset per rule #6 above), assigns every entity/relation a consistent integer id (train-vocab-first, so ids are reproducible), and bundles train/valid/test into a `KGDataset`. Downloads the dataset if missing (verified working against real FB15k-237 — stats matched published numbers exactly); falls back to a small fully-synthetic fake graph if every download mirror fails, purely so the id-mapping logic itself can still be tested offline. On-disk column order `(head, relation, tail)` confirmed correct — no repeat of WN18's column-order bug. |
| `preprocessing/graph_builder.py` | Builds the message-passing edge list from **training triples only** (never valid/test — this is non-negotiable rule #3), adding inverse edges for bidirectional message flow. `assert_no_leakage()` is a sanity check that fails loudly if validation/test data ever leaks into the graph or if any triple duplicates across splits. |
| `preprocessing/toy_subset.py` | Carves a small, real, connected chunk (default 50 entities) out of the full training graph via breadth-first search, then remaps its entity ids to a fresh contiguous range so the subset is fully self-consistent. Exists so later phases can be smoke-tested in seconds instead of waiting on the full ~272k-triple FB15k-237 training set. |
| `experiments/configs/phase0_fb15k237.yaml` | Run settings for Phase 0 (dataset location, download/fallback behavior, toy-subset size, seed) — kept out of code so they can change without editing Python. |
| `run_phase0_smoke_test.py` | The Phase 0 gate script. Loads the config, loads the dataset, and runs every check above end-to-end, printing `PASSED` or a specific failure. |
| `data/raw/`, `data/processed/` | Empty, gitignored directories where the real dataset and any derived files land — not checked into version control. |
| `models/kg_encoder.py` | `RGCNEncoder` — a stack of PyTorch Geometric `RGCNConv` layers (R-GCN chosen as the simplest relation-aware encoder to debug, per the Phase 1 plan; this is the project's primary Phase 1 encoder). Takes entity features + edge_index + edge_type, returns structure-aware entity embeddings. |
| `models/kg_encoder_rgat.py` | `RGATEncoder` — a second Phase 1 encoder variant with the same shape contract as `RGCNEncoder`. Toy gate reported passed after the dead-parameter fix; revised full-dataset run remains unverified. |
| `models/scorer.py` | `DistMultScorer` — the KGE scoring layer (DistMult first, per rule #5/README §3.1). Scores single triples and, for evaluation, scores one triple against every entity at once (`score_all_tails`/`score_all_heads`). |
| `models/kg_only_baseline.py` | `KGOnlyBaseline` — composes an entity embedding table + `RGCNEncoder` + `DistMultScorer` into the actual Phase 1 model. This is also the literal "KG-only baseline" row in the final validation table (README.md §12), not just a stepping stone. |
| `models/kg_only_baseline_rgat.py` | `KGOnlyBaselineRGAT` — same composition as `KGOnlyBaseline` but with `RGATEncoder` instead of `RGCNEncoder`. Selected via `model.encoder_type: rgat` in a config (see `training/model_factory.py`), not used by default. |
| `training/model_factory.py` | `build_model()` — the one place that maps a config's `model.encoder_type` (`"rgcn"` default, or `"rgat"`) to the right model class. Used by both `training/train_kg_baseline.py` and the Phase 1 smoke-test scripts so they can't drift out of sync on how a config becomes a model. |
| `training/losses.py` | `bce_loss` — BCE/logistic loss matched to DistMult's bilinear scorer (non-negotiable rule #2 — never margin-based ranking loss here). |
| `training/negative_sampling.py` | `sample_negatives` — corrupts head or tail per positive triple, filtered against **training-only** known-true triples (a narrower set than evaluation's filter index — see the module docstring for why the two must not be conflated). |
| `evaluation/metrics.py` | `build_filter_index` + `evaluate_filtered` — standard filtered-ranking MRR/Hits@1/3/10, filtering built from all splits combined (used only for ranking, never as training signal). |
| `training/train_kg_baseline.py` | The Phase 1 training loop. Config-driven so the same code runs the toy-subset smoke test and the full FB15k-237 run for *either* encoder variant (via `training/model_factory.py`) — encodes the whole graph once per optimizer step and reuses it for every positive/negative triple in that step (rule #7), rather than recomputing per triple. |
| `experiments/configs/phase1_toy.yaml` | Phase 1 toy-subset run settings, R-GCN (dim=32, 20 epochs — deliberately tiny/fast). **Verified in Colab — gate passed.** |
| `experiments/configs/phase1_rgat_toy.yaml` | Same toy-subset settings as `phase1_toy.yaml`, but `model.encoder_type: rgat`. **Verified in Colab — gate passed.** |
| `run_phase1_smoke_test.py` | The Phase 1 toy-subset gate script for R-GCN (its default config), mirroring `run_phase0_smoke_test.py`'s pattern: trains on the toy subset and checks loss is finite and improves, validation metrics compute without error, and the saved checkpoint reloads with identical parameters (compared by parameter *name*, not position — reports the exact mismatched name(s)/diff if this ever fails, per the RGAT `l2` bug this caught). Now accepts an optional config-path argument to `main()`, reused by the RGAT variant below. **Passed in Colab (R-GCN).** |
| `run_phase1_rgat_smoke_test.py` | Same checks as `run_phase1_smoke_test.py` (imports and reuses its `main()`), pointed at `phase1_rgat_toy.yaml` instead — its own entry point so the RGAT variant has the same one-command gate. **Passed in Colab.** |
| `experiments/configs/phase1_full.yaml` | Real FB15k-237 training settings, R-GCN (dim=128 — see Phase 1 notes below for why this was lowered from the originally-planned 256 — 100 epochs, `batch_size: 32768`, `save_best: true`, `use_synthetic_fallback: false` since a silent fallback here would be misleading). **Run in Colab — completed successfully** (see Phase 1 notes below for the full result). |
| `run_phase1_full_training.py` | Launches the real Phase 1 training run using `phase1_full.yaml` by default, or another config path passed as `sys.argv[1]`. Use this rather than invoking `training/train_kg_baseline.py` directly — running that file as a bare script puts its own folder, not the repo root, on `sys.path`, breaking its `models`/`evaluation`/`preprocessing` imports (see this script's own docstring). Produces `experiments/checkpoints/kg_only_baseline.pt`. |
| `experiments/configs/phase1_rgat_full.yaml` | Real FB15k-237 training settings, RGAT — `dim: 32`, `heads: 1` (dropped from a first attempt at `dim=128`/`heads=2` copied from R-GCN, which hit a 66.44 GiB CUDA OOM in `RGATConv`'s per-edge weight gather — see Phase 1 notes below for why this scales completely differently from R-GCN's OOM and can't just reuse R-GCN's fix). **Not yet run with the fixed config.** |
| `run_phase1_rgat_full_training.py` | Defaults to `phase1_rgat_full.yaml`, targeting `experiments/checkpoints/kg_only_baseline_rgat.pt`. Initial full attempt failed; revised-config completion is not recorded. |
| `models/kg_lm_projection.py`, `models/lm_kg_projection.py` | Trainable projections using Linear, LayerNorm, GELU and Dropout; configurable KG dimension. |
| `models/frozen_lm.py`, `models/kg_lm_bridge.py` | Frozen RoBERTa soft-prompt conditioning and complete standalone KG-to-LM-to-KG bridge. |
| `preprocessing/entity_text.py`, `preprocessing/relation_text.py` | Text acquisition, cleaning, fallback and alignment to existing KG IDs. |
| `experiments/configs/phase2_lm_toy.yaml`, `run_phase2_smoke_test.py` | Generic standalone bridge gate with dummy KG vectors. |
| `experiments/configs/phase3_integration.yaml` | Selected R-GCN/128 integration contract, matching the full baseline with a 128-dimensional bridge. Phase 3 runner and checkpoint-backed gate remain pending. |
| `run_phase2_real_text_smoke_test.py`, `run_phase2_relation_text_smoke_test.py` | Real-text entity/relation gates with dummy KG vectors; local/T4 passes reported. Settings are currently constants in each script. |
| `run_entity_text_alignment_check.py`, `run_long_text_alignment_check.py`, `run_relation_text_alignment_check.py` | Entity and relation text alignment diagnostics. |
| `experiments/RUNS.md` | Evidence-aware run register and checkpoint recovery inventory. |
| `experiments/checkpoints/` | Gitignored intended output directory; no historical weights recovered locally. Phase 1 save claims and missing-source details are recorded in experiments/RUNS.md. |

## Current stage / priority

**Phase 2 complete; Phase 3 next.** This is the current project status as of
2026-10-05, superseding older planning text that placed Phase 2 in the future.

- Phase 0: real FB15k-237 data gate reported passed in Colab.
- Phase 1: R-GCN toy/full and RGAT toy gates reported passed. The R-GCN full
  run reports validation MRR 0.1842 and Hits@10 0.3409 at epoch 100; these are
  not test results. The revised RGAT full config (dimension 32, one head) has
  no recorded successful full run.
- Phase 2: standalone entity/relation semantic bridge implemented; local/T4
  passes and full text alignment coverage reported. Inputs are still dummy
  structural vectors, not connected Phase 1 outputs.
- Artifacts: historical checkpoints and original logs have not been recovered
  locally. Consult [the recovery inventory](experiments/RUNS.md#checkpoint-recovery-inventory)
  before treating a configured checkpoint path as available. Exact historical
  run revisions and external storage locations remain unknown.

The encoder/dimension decision is complete: **R-GCN, dimension 128**, with two
layers, 30 bases and dropout 0.2, matching the reported full baseline.
Next implementation work is Phase 3: recover/validate `kg_only_baseline.pt`
and integrate real entity and relation embeddings with their aligned text
bridge at dimension 128. Check finite, nonzero upstream gradients while keeping
the LM frozen. The historical dimension-32 smoke tests remain separate evidence;
the selected 128-dimensional checkpoint-backed integration gate is still pending.

Then add graph encoder 2 (Phase 4), assemble full DistMult training/evaluation
(Phase 5), and run comparisons, ComplEx and ablations (Phase 6). A standalone
bridge smoke-test pass is not evidence of improved link prediction.

The earlier WN18RR diagnostic results remain excluded from valid comparisons
because they predate the project's dataset and loss corrections. See
[README.md](README.md#diagnostic-history-and-limitations).
