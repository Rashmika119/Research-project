# Structural and textual knowledge-graph learning

Research code for **Integrating Structural and Textual Semantics for
Ontology-Enriched Knowledge Graph Representation Learning**, University of
Kelaniya. The task is filtered head/tail link prediction on FB15k-237. The
implemented scope is relational triples, not formal OWL/description-logic reasoning.

## Start in Google Colab

Open [notebooks/research_pipeline.ipynb](notebooks/research_pipeline.ipynb).
Select a GPU runtime, set the repository ref to a pushed version containing these
changes, and run the cells in order. Defaults train a full independent graph-only
baseline and fifteen scratch pilot experiments. The optional final full-data
experiment is disabled with `RUN_FINAL_FULL_DATASET = False`.

The notebook has 16 labeled sections covering setup, Drive, data validation,
offline/real-LM smoke tests, training, reports, selection, plots, and final export.
It stores data, Hugging Face downloads, pooled-text caches, checkpoints, configs,
histories, results, CSVs, and PNG plots under Google Drive. Local repository files
alone do not preserve Colab runtime state.

## Current experiment protocol

**No graph-only baseline weights initialize any of the five scratch variants.**
The independent baseline is a comparison artifact only. Scratch means fresh
entity/relation tables, graph encoders, projections, and gates. The frozen
`roberta-base` encoder intentionally retains its pretrained language weights.

| Stage | Data | Budget | Initialization |
| --- | --- | --- | --- |
| Independent RGAT + ComplEx baseline | Full FB15k-237 | 100 epochs; validate every 5 | Random task weights |
| Architecture pilot | Fixed subset, at most 1,000 entities | 5 variants × seeds 0/1/2 × 30 total epochs | New model for every run |
| Optional selected architecture | Full original dataset | Default 100 total epochs | New model; no pilot/baseline weights |

The full data has 14,541 entities, 237 relations, 272,115 training facts,
17,535 validation facts, and 20,466 test facts. Research runners validate these
counts and disable synthetic fallback. A synthetic graph is used only in tests.

All models use width 32 (16 real + 16 imaginary ComplEx coordinates), fixed Adam
LR 0.001, weight decay 0.00001, four negative samples per positive, BCE loss,
gradient clipping at 1.0, and the existing exact-tie average-rank evaluator.
Scoring mathematics, RGAT implementation, and dataset splitting are unchanged.
The baseline uses batches of 32,768 positive triples. Pilot runs use the complete
sampled training set per update: **one optimizer step per epoch**.

### Structural warmup and the 30-epoch budget

Historically, "warmup" meant loading a separately trained full-data graph model.
There was no independent warmup loop or learning-rate warmup scheduler. That
baseline dependency is removed from the new protocol.

The user selected **5 structural-only epochs followed by 25 integrated epochs**
for `original`, `residual`, `no-refinement`, and `residual-no-softprompt`.
`original-no-warmup` runs **30 integrated epochs**. Every pilot therefore has
30 optimizer updates. Warmup is included, never added to the requested budget.

All components are initialized before warmup. During warmup, only the structural
entity table, first RGAT, and relation table receive updates. The same Adam
optimizer continues into integrated training, retaining its structural moments.
LR stays fixed. Newly added modules acquire optimizer state when they first
receive gradients. Original/no-warmup have identical architecture and initial
parameters for a paired seed; their update objectives/schedules differ.

Warmup validation scores are explicitly labeled structural diagnostics and never
select an integrated checkpoint. The integrated model is evaluated immediately
after warmup, before its first integrated update. That boundary checkpoint can
remain best. For no-warmup this is epoch 0; with five warmup epochs it is total
epoch 5 / integrated epoch 0. `best_epoch`, `best_main_epoch`, actual warmup/main
epochs, and optimizer-step counts are recorded.

### Five architectures

Let E be first-RGAT entity vectors, R the learned ComplEx relation table, B the
shared entity/relation soft-prompt bridge, and G2 the second RGAT. Both graph
encoders are two-layer modules; relations enter scoring through their own text path.

| Variant | Entity representation | Relation representation | Warmup |
| --- | --- | --- | --- |
| `original` | `G2(B(E, text), A)` | `B(R, text)` | 5 |
| `residual` | `U + tanh(gF) * G2(U, A)` | `R + tanh(gR) * TR` | 5 |
| `no-refinement` | `E + tanh(gE) * TE` | `R + tanh(gR) * TR` | 5 |
| `original-no-warmup` | Exactly the original architecture | Exactly the original architecture | 0 |
| `residual-no-softprompt` | Same residual equations and second RGAT | Same gated relation fusion | 5 |

For residual models, `U = E + tanh(gE) * TE`. The three scalar gate parameters
start at 0.01; no-refinement has only the entity/relation gates.

Soft-prompt variants project structural vectors 32→768, prepend the resulting
continuous prompt token to descriptions, read RoBERTa's contextualized prompt
position, and project 768→32. Both projections use Linear, LayerNorm, GELU, and
Dropout. RoBERTa stays frozen and in evaluation mode, while gradients through its
operations reach the input projection and structural model.

`residual-no-softprompt` has **no KG→LM projection or structural LM input**.
It calls RoBERTa with description token IDs and attention masks only, then computes
`sum(mask * hidden) / sum(mask)` over unmasked tokens, including tokenizer special
tokens. Padding is excluded; an all-masked sequence is rejected. A trainable
768→32 projection produces TE/TR. The second RGAT and refinement shortcut remain.

Only frozen pooled 768-wide outputs are cached. Content-addressed keys include
ordered tokens/masks, model identity/revision, pooling convention, relevant library
versions and device type. The projection executes on every step and receives
gradients. Soft-prompt outputs cannot be cached because they depend on changing KG
weights. Default text length is 64 tokenizer positions, plus one prompt position
where applicable. Default text chunk size is 4; chunks use activation checkpointing
for the soft-prompt training path.

## Commands

Run from the repository root after `pip install -r requirements.txt`.
Replace output paths below with mounted Drive paths in Colab.

```bash
# Initial offline gates (tiny LM substitute; real RGAT and ComplEx)
python run_tie_ranking_check.py
python run_complex_scorer_check.py
python run_scratch_smoke_test.py --offline-only --device cpu

# Additional gate using real pretrained RoBERTa; downloads it if uncached
python run_pretrained_check.py --device cuda
python run_scratch_smoke_test.py --real-lm-only --device cuda

# Independent exact CUDA checkpoint checks, followed by real-LM training/resume
python run_resume_diagnostic.py --device cuda --repeats 3 --output-dir experiments/research/resume_check
python run_resume_diagnostic.py --device cuda --real-lm --variant residual --repeats 1 --output-dir experiments/research/real_resume_check

# Independent full-data baseline: no LM, with held-out test evaluation
python run_research_baseline.py --output-dir experiments/research/baseline

# One scratch pilot run; no --checkpoint argument
python run_phase5_training.py --variant residual --max-entities 1000 --epochs 30 --warmup-epochs 5 --output-dir experiments/research/residual_seed0

# Fifteen independent runs and automatic JSON/CSV/PNG reporting
python run_phase5_comparisons.py --output-dir experiments/research/pilot --skip-completed --resume

# Example full-data run AFTER validation selection; substitute the actual winner
python run_phase5_training.py --variant residual --max-entities 0 --epochs 100 --warmup-epochs 5 --eval-every 5 --output-dir experiments/research/final_residual
```

`experiments/configs/research_protocol.yaml` provides notebook stage settings.
The CLI has matching defaults; changing YAML alone does not change CLI defaults.
Use `--help` for explicit CLI overrides. `--max-entities 0` means all original
training triples and original full held-out splits, not a sampled graph.

### Artifacts, skipping and resuming

Each run saves:

```text
config.json          # experiment config, data/source identity, environment
manifest.json        # exact entity/relation maps and train/valid/test facts
best.pt              # validation-selected weights; frozen LM omitted
last.pt              # latest epoch, optimizer/RNG state, selected weights
resume_audit.json     # exact model, optimizer, RNG and mode checks before continuing
history.json/.csv    # epoch, phase, loss, validation, timing and update counts
metrics.csv          # selected checkpoint validation/test metrics
results.json         # machine-readable completed result and checkpoint checksums
```

`--skip-completed` verifies configuration, dataset/subset/token identity, code,
environment and checkpoint checksums before skipping. `--resume` restores only
that output directory's `last.pt`, including optimizer and random states. The
baseline and other experiment directories cannot be used as initialization inputs.
Changed settings/data/code/environment require a new directory. Atomic temporary
file replacement protects epoch checkpoints; `last.pt` is the commit point and
also carries the best weights for recovery between best/last writes.

The notebook stores verified RoBERTa weights in `/content/hf_cache/hub`, with a
verified backup on Drive. It validates the pinned revision and file hashes before
loading and repairs only affected cached files. A mandatory pretrained preflight
blocks experiments on failure. Separate CPU regressions, GPU real-RoBERTa checks,
and CUDA resume cells save complete logs on Drive. Use a new run directory for
the updated checkpoint schema. See [reliability notes](docs/RELIABILITY.md) for
failure diagnosis, cache handling, exact state checks and verification results.

The comparison adds a shared subset manifest, `comparison_config.json`, `runs.json`,
`runs.csv`, `summary.json`, `comparison.csv`, `training_curves.png`, and
`validation_comparison.png`. Runs execute sequentially in subprocesses. The report
requires every expected variant/seed and verifies matching subsets, optimization
settings, paired structural initialization and ranking policies.

### Evaluation and interpretation

Validation selects each checkpoint. The best architecture is selected by mean
validation MRR across seeds 0/1/2; sample standard deviation is reported. Test MRR
and Hits@1/3/10 are descriptive, never selection inputs. All exact-tied candidates
receive average rank; optimistic MRR and tie statistics remain diagnostics.

The existing connected subset sampler retains original held-out triples whose
endpoints are selected. Its sampled training graph need not contain every original
training fact among those entities. Negative sampling and evaluation filtering
therefore include **all original training facts covered by the candidate set**.
Only sampled training facts become pilot message edges. Validation/test facts are
used only by evaluation and its filtered-candidate index.

Do not compare the full-data baseline MRR against pilot MRR: candidate sets differ.
The optional final export accepts only results on the same full dataset. It records
training budgets: the baseline has about nine updates per epoch, while the default
full integrated model has one. Same epochs do not mean equal compute/update budgets.
The final default is one seed, so it does not establish full-data multi-seed variance.

Comparing original/residual changes multiple residual paths. Comparing residual/
no-refinement tests the additional refinement branch. Comparing residual/
residual-no-softprompt changes structural conditioning AND representation pooling;
the no-softprompt model also has fewer trainable projection parameters. Interpret
it as the requested encoding-mechanism comparison, not an isolated token-deletion
experiment. No performance improvement is assumed.

## Compatibility and historical research

The original Phase 0–4 gates, old baseline trainers, text checks, scoring checks,
and historical checkpoint reevaluator remain available. For explicitly reproducing
the old pretrained workflow only:

```bash
python run_phase5_training.py --legacy-pretrained --checkpoint PATH_TO_OLD_BASELINE --variant original --output-dir experiments/phase5_legacy_reproduction
```

This mode lives in `training/legacy_phase5.py`; the notebook and new comparison
runner never invoke it. Legacy reevaluation remains `run_reevaluate_checkpoint.py`.
It targets the historical checkpoint format. New checkpoints are loaded/resumed
by the research runner. Do not mix their experiment populations in one table.

[Historical notes](docs/HISTORICAL_NOTES.md) preserve the old CLAUDE.md verbatim,
including user-reported Colab results and superseded plans. Those historical metrics
used optimistic ties unless explicitly labeled otherwise. They are not results of
the new scratch protocol.

## Validation and limits

See [implementation notes](docs/IMPLEMENTATION.md) for changed files, test coverage,
and cleanup decisions. Offline regression tests use real RGAT/ComplEx with a small
deterministic substitute for RoBERTa; these cannot establish real-LM GPU feasibility.
The real-LM gate passed locally on CPU for all five architectures using cached
pretrained RoBERTa and synthetic facts. CUDA/full-data feasibility remains untested.
No expensive full research experiment has been run to validate these code changes,
and no new benchmark result is claimed.

Full-data two-RGAT plus soft-prompt RoBERTa may exceed Colab GPU memory/runtime.
Lower text/evaluation batch sizes can help their respective allocations; they do
not remove full-graph RGAT activation costs. There is no silent downsampling,
dimension reduction, mixed precision, or objective substitution. Fixed seeds and
saved random states and strict deterministic algorithms support repeatability
within the same environment. Unsupported deterministic kernels fail explicitly;
cross-device/version equality is not promised. Per-run package versions, numerical
policy and source hashes are recorded. CUDA acceptance still requires the GPU gates.
