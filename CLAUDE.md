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
6. **WN18 is this project's primary dataset, by explicit supervisor
   direction — not WN18RR.** This is a deliberate deviation from the more
   commonly cited modern benchmark: WN18 contains reversible/redundant
   relations (e.g. `_hypernym`/`_hyponym` mirror pairs) that let a model score
   well by exploiting inverse-relation shortcuts rather than learning real
   structure — which is exactly why WN18RR was created in the first place.
   Given that, when interpreting or reporting results:
   - Don't compare our WN18 MRR/Hits@K numbers directly against published
     WN18RR numbers in other papers — they're not the same task difficulty,
     and a WN18 score will typically look better for reasons unrelated to
     model quality.
   - Note this WN18-vs-WN18RR choice explicitly wherever results are written
     up, so it reads as a stated methodological decision, not an oversight.
   - Don't silently swap datasets between comparable runs regardless — every
     run in the validation table (README.md §12) should use the same
     dataset unless the comparison is specifically about dataset choice.
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

1. **Phase 0 — Data & config scaffolding. ✅ Implemented, gate not yet run.**
   Entity/relation ID mapping, train/val/test split, adjacency structures
   built from *training triples only*, a config file holding the run
   settings, and a small connected toy subset for smoke testing later
   phases. See "Repository Map" below for the exact files.
   *Gate:* splits don't overlap, adjacency has no val/test edges, config
   loads and round-trips — checked by `run_phase0_smoke_test.py`, which
   has been written but not yet executed (no local Python interpreter in
   this dev environment; needs to be run in Colab or any machine with
   Python + `requirements.txt` installed). **Do not start Phase 1 model
   code until this has actually been run and printed PASSED.**

2. **Phase 1 — KG-only baseline (Encoder Phase 1 + scorer, no LM at all).**
   Implement the relation-aware encoder and DistMult scorer as a standalone
   trainable model. Smoke-test on the toy subset first (loss decreases,
   shapes correct, no NaNs), then train on the real filtered dataset and
   check validation MRR/Hits@K land in a plausible range for the
   scorer/dataset (sanity-check against known KGE literature ballparks).
   Save a checkpoint (`kg_only_baseline.pt`) — this becomes a reusable
   artifact and the actual "KG-only baseline" row in the validation table,
   not just a stepping stone.
   *Gate:* stable training curve, reasonable filtered MRR/Hits@K, checkpoint
   saved and reloadable.

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
| `preprocessing/dataset.py` | Reads WN18's raw text triples (primary dataset per rule #6 above, not WN18RR), assigns every entity/relation a consistent integer id (train-vocab-first, so ids are reproducible), and bundles train/valid/test into a `KGDataset`. Downloads the dataset if missing (URL unverified — see in-file note); falls back to a small fully-synthetic fake graph if there's no network, purely so the id-mapping logic itself can still be tested offline. |
| `preprocessing/graph_builder.py` | Builds the message-passing edge list from **training triples only** (never valid/test — this is non-negotiable rule #3), adding inverse edges for bidirectional message flow. `assert_no_leakage()` is a sanity check that fails loudly if validation/test data ever leaks into the graph or if any triple duplicates across splits. |
| `preprocessing/toy_subset.py` | Carves a small, real, connected chunk (default 50 entities) out of the full training graph via breadth-first search, then remaps its entity ids to a fresh contiguous range so the subset is fully self-consistent. Exists so later phases can be smoke-tested in seconds instead of waiting on the full ~87k-triple dataset. |
| `experiments/configs/phase0_wn18.yaml` | Run settings for Phase 0 (dataset location, download/fallback behavior, toy-subset size, seed) — kept out of code so they can change without editing Python. |
| `run_phase0_smoke_test.py` | The Phase 0 gate script. Loads the config, loads the dataset, and runs every check above end-to-end, printing `PASSED` or a specific failure. **This is the file to actually run** before writing any Phase 1 model code. |
| `data/raw/`, `data/processed/` | Empty, gitignored directories where the real dataset and any derived files land — not checked into version control. |

## Current stage / priority

Intermediate report stage is complete. Phase 0 (this repo's own data
scaffolding, distinct from the report's research-methodology phases) is
implemented but **not yet verified** — `run_phase0_smoke_test.py` has not
been executed anywhere yet. Next actual work, in order:
1. Run `run_phase0_smoke_test.py` (in Colab or any Python environment) and
   confirm it prints PASSED. Fix anything it flags before moving on.
2. Phase 1 — stable KG-only baseline (correct BCE-style objective, tuned
   hyperparameters), built and smoke-tested per the phased plan above.
3. Full KG → LM → KG pipeline implementation per the build order above.
4. Baseline comparison (KG-only vs text-enhanced vs proposed w/ DistMult vs
   proposed w/ ComplEx).
5. Ablations (warm-up on/off, embedding dimension).

Do not present WN18RR results from the preliminary experiment as evidence
about the proposed architecture's viability — they predate loss-function and
dataset corrections and are diagnostic only (README.md §31).
