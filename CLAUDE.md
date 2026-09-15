# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Research codebase for a BSc (Hons) Software Engineering intermediate research
project at the University of Kelaniya: **"Integrating Structural and Textual
Semantics for Ontology-Enriched Knowledge Graph Representation Learning."**

Full narrative context (research questions, literature review, methodology,
diagnostic experiment results, limitations) lives in [README.md](README.md) —
read it before making architectural decisions. This file only covers the
implementation rules Claude needs when writing or editing code here.

Phase 0 (data scaffolding) is implemented — see "Repository Map" below for
what exists and why. `models/`, `training/`, and `evaluation/` don't exist
yet; those start in Phase 1. Check with Glob before assuming structure beyond
what's listed here, in case it's changed since this file was last updated.

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
   **Not yet run** — `run_phase1_rgat_smoke_test.py` needs to pass in
   Colab (CPU is fine, same as the R-GCN toy gate) before a real
   full-dataset RGAT run is attempted. RGAT is *at least* as likely as
   R-GCN was to hit GPU memory/speed issues at full scale (attention adds
   more per-relation state on top of R-GCN's already-expensive per-relation
   loop) — expect to revisit `dim`/`batch_size`/`num_bases` again for it
   specifically, don't assume the R-GCN full-run config's values transfer.

3. **Phase 2 — LM module in isolation.**
   Build the KG→LM projection, frozen-LM wrapper, soft-prompt injection, and
   LM→KG projection as their own standalone module — tested with *random/dummy*
   KG vectors, not real Phase 1 output yet. Verify: projected shapes are
   correct at both boundaries (e.g. 256→768 and 768→256), the LM's parameters
   truly have no gradient after a backward pass, a forward pass fits in Colab
   GPU memory at the intended batch size, and output is deterministic for a
   fixed input/seed.
   *Gate:* shape assertions pass, confirmed zero gradient on frozen LM params,
   memory profile acceptable.

4. **Phase 3 — Integrate Phase 1 → Phase 2 (KG → LM).**
   Feed real Phase 1 embeddings (loaded from the Phase 1 checkpoint) into the
   now-validated LM module. Run a small forward/backward smoke test and
   confirm gradients flow into Encoder Phase 1 and the projection layers but
   not into the frozen LM.
   *Gate:* integrated forward/backward runs cleanly on the toy subset with no
   shape or gradient-flow surprises.

5. **Phase 4 — Add KG Encoder Phase 2 (LM → KG refinement).**
   Build Encoder Phase 2 standalone first (dummy LM-shaped input vectors),
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

This is the standard sequence for running any phase's script/notebook in a
fresh Colab runtime. Right now that means `run_phase0_smoke_test.py`; later
phases will follow the same pattern with their own entry-point script.

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

Everything below is Phase 0 output — data plumbing only, no model code yet.

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
| `models/kg_encoder_rgat.py` | `RGATEncoder` — a second Phase 1 encoder variant, a stack of PyTorch Geometric `RGATConv` layers (relation-aware attention on top of R-GCN's per-relation weight matrices). Same shape contract/interface as `RGCNEncoder` — a drop-in swap. Built for a later encoder comparison, not as a replacement (see Phase 1 notes below). **Not yet smoke-tested.** |
| `models/scorer.py` | `DistMultScorer` — the KGE scoring layer (DistMult first, per rule #5/README §3.1). Scores single triples and, for evaluation, scores one triple against every entity at once (`score_all_tails`/`score_all_heads`). |
| `models/kg_only_baseline.py` | `KGOnlyBaseline` — composes an entity embedding table + `RGCNEncoder` + `DistMultScorer` into the actual Phase 1 model. This is also the literal "KG-only baseline" row in the final validation table (README.md §12), not just a stepping stone. |
| `models/kg_only_baseline_rgat.py` | `KGOnlyBaselineRGAT` — same composition as `KGOnlyBaseline` but with `RGATEncoder` instead of `RGCNEncoder`. Selected via `model.encoder_type: rgat` in a config (see `training/model_factory.py`), not used by default. |
| `training/model_factory.py` | `build_model()` — the one place that maps a config's `model.encoder_type` (`"rgcn"` default, or `"rgat"`) to the right model class. Used by both `training/train_kg_baseline.py` and the Phase 1 smoke-test scripts so they can't drift out of sync on how a config becomes a model. |
| `training/losses.py` | `bce_loss` — BCE/logistic loss matched to DistMult's bilinear scorer (non-negotiable rule #2 — never margin-based ranking loss here). |
| `training/negative_sampling.py` | `sample_negatives` — corrupts head or tail per positive triple, filtered against **training-only** known-true triples (a narrower set than evaluation's filter index — see the module docstring for why the two must not be conflated). |
| `evaluation/metrics.py` | `build_filter_index` + `evaluate_filtered` — standard filtered-ranking MRR/Hits@1/3/10, filtering built from all splits combined (used only for ranking, never as training signal). |
| `training/train_kg_baseline.py` | The Phase 1 training loop. Config-driven so the same code runs the toy-subset smoke test and the full FB15k-237 run for *either* encoder variant (via `training/model_factory.py`) — encodes the whole graph once per optimizer step and reuses it for every positive/negative triple in that step (rule #7), rather than recomputing per triple. |
| `experiments/configs/phase1_toy.yaml` | Phase 1 toy-subset run settings, R-GCN (dim=32, 20 epochs — deliberately tiny/fast). **Verified in Colab — gate passed.** |
| `experiments/configs/phase1_rgat_toy.yaml` | Same toy-subset settings as `phase1_toy.yaml`, but `model.encoder_type: rgat`. **Not yet run.** |
| `run_phase1_smoke_test.py` | The Phase 1 toy-subset gate script for R-GCN (its default config), mirroring `run_phase0_smoke_test.py`'s pattern: trains on the toy subset and checks loss is finite and improves, validation metrics compute without error, and the saved checkpoint reloads with identical parameters. Now accepts an optional config-path argument to `main()`, reused by the RGAT variant below. **Passed in Colab (R-GCN).** |
| `run_phase1_rgat_smoke_test.py` | Same checks as `run_phase1_smoke_test.py` (imports and reuses its `main()`), pointed at `phase1_rgat_toy.yaml` instead — its own entry point so the RGAT variant has the same one-command gate. **Not yet run.** |
| `experiments/configs/phase1_full.yaml` | Real FB15k-237 training settings, R-GCN (dim=128 — see Phase 1 notes below for why this was lowered from the originally-planned 256 — 100 epochs, `batch_size: 32768`, `save_best: true`, `use_synthetic_fallback: false` since a silent fallback here would be misleading). **Run in Colab — completed successfully** (see Phase 1 notes below for the full result). |
| `run_phase1_full_training.py` | Launches the real Phase 1 training run using `phase1_full.yaml`. Use this rather than invoking `training/train_kg_baseline.py` directly — running that file as a bare script puts its own folder, not the repo root, on `sys.path`, breaking its `models`/`evaluation`/`preprocessing` imports (see this script's own docstring). Produces `experiments/checkpoints/kg_only_baseline.pt`. |
| `experiments/checkpoints/` | Empty, gitignored directory where trained model checkpoints land (`phase1_toy.pt` and `kg_only_baseline.pt` already produced there; `phase1_rgat_toy.pt` once the RGAT toy gate is run). |

## Current stage / priority

Intermediate report stage is complete. Phase 0 is **done — verified in
Colab against real FB15k-237**, gate passed (see rule #6 for the WN18RR →
WN18 → FB15k-237 dataset history). Phase 1 is **done — both toy-subset and
full-dataset gates passed in Colab**: the real 100-epoch run on FB15k-237
completed with a stable loss curve and validation MRR/Hits@10 rising
throughout (final val_MRR=0.1842, val_Hits@10=0.3409 — see Phase 1's "Real
run" notes above for the full curve and honest context on how this compares
to published numbers), checkpoint saved to
`experiments/checkpoints/kg_only_baseline.pt`. This is the real "KG-only
baseline" row for the validation table — not just a smoke-test artifact.
Note metrics were still improving at epoch 100 (not plateaued), so more
epochs / tuning (num_negatives, dim back up to 256 now that the OOM and
speed issues are understood) could strengthen this baseline later if
desired — optional, not a blocker. An RGAT encoder variant
(`models/kg_encoder_rgat.py`, `models/kg_only_baseline_rgat.py`) was also
built alongside R-GCN, by explicit user decision, for a later encoder
comparison — R-GCN remains the primary baseline; RGAT's own toy-subset gate
(`run_phase1_rgat_smoke_test.py`) has **not yet been run**. Next actual
work, in order:
1. Run `run_phase1_rgat_smoke_test.py` in Colab (CPU is fine) to gate-test
   the new RGAT variant, same phased-build discipline as everything else —
   toy smoke test before any real run.
2. Full KG → LM → KG pipeline implementation per the build order above
   (Phases 2–5) — starting with Phase 2 (LM module in isolation), now that
   the R-GCN baseline gives us a trustworthy Phase 1 checkpoint to
   eventually feed into it. (Not blocked on step 1 above — the RGAT
   variant is a parallel comparison track, not a prerequisite for Phase 2.)
3. Baseline comparison (KG-only vs text-enhanced vs proposed w/ DistMult vs
   proposed w/ ComplEx) — and, informally, R-GCN vs RGAT as encoders.
4. Ablations (warm-up on/off, embedding dimension).

Do not present WN18RR results from the preliminary experiment as evidence
about the proposed architecture's viability — they predate loss-function and
dataset corrections and are diagnostic only (README.md §31).
