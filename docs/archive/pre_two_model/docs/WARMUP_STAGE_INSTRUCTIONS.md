# CLAUDE.md - current project instructions

Read README.md, experiments/configs/research_protocol.yaml and
 docs/WARMUP_ABLATION_5000.md for the active stage. Completed-stage protocol and
user-reported results are preserved in docs/STAGE1_PROTOCOL.md and
 docs/STAGE1_RESULTS.md. Older notes in docs/HISTORICAL_NOTES.md are historical,
including superseded architecture decisions and optimistic-tie metrics.

## Active research protocol

Compare exactly four scratch configurations on the same 5,000-entity FB15k-237
subset, subset seed 0, training seeds 0/1/2 (12 independent runs):

- residual-warmup: residual architecture, 5 structural + 25 integrated epochs.
- residual-no-warmup: identical residual architecture, 0 + 30 epochs.
- residual-no-softprompt-warmup: independent-text residual, 5 + 25 epochs.
- residual-no-softprompt-no-warmup: identical independent-text architecture, 0 + 30.

Separate architecture selection from warmup selection. Available architectures are
residual, residual-no-softprompt and no-refinement. No-Refinement remains usable
but is excluded from this study. Original and Original No-Warmup must not return
to active registrations, CLI choices or comparison selection. Keep shared classes
needed by explicit historical workflows; do not delete historical artifacts.
The independent full-data RGAT + ComplEx baseline remains available separately.

All task-specific modules initialize from scratch before warmup. Never initialize
from another configuration, seed, baseline or historical checkpoint. Only pretrained
frozen RoBERTa weights may be reused. Same-architecture warmup pairs must have
identical initial complete trainable-state fingerprints for the same seed. Warmup
retains only its own structural parameters and Adam moments. It is structural BCE
training, not an LR schedule. Use one full-subset optimizer update per epoch:
5 structural + 25 integrated or 30 integrated updates, 30 total in both cases.

Preserve width 32, two-layer/one-head RGATs, Adam LR .001, weight decay .00001,
four negatives, BCE, gradient clipping 1, full precision, text length 64, text
batch 4 and evaluation batch 64. Validate each epoch; select by mean validation
MRR across seeds, never test. Report sample SD and descriptive paired differences;
three seeds do not establish statistical significance. Test data is untouched
until evaluating the validation-selected checkpoint.

## Dataset rules

Use train_bfs_induced_v1: deterministic training-only BFS entity selection, then
retain every original training fact between selected endpoints. Fixed ordered
training facts and subset seed determine selection. If a component is exhausted,
choose another root from remaining sorted training-incident entity IDs. Fail if
exact requested count is impossible; never silently reduce entities or select by
held-out edges. Preserve original valid/test split boundaries, relation IDs and
sorted-original-ID entity remapping. Message edges are training facts and inverses
only. Negative sampling rejects original training facts in candidate scope, never
uses held-out labels. Filtering held-out positives is evaluation-only.

Persist and verify the shared versioned manifest, mappings, exact splits, full-data
and subset hashes, graph counts/connectivity/density. Every run uses the same
5,000 ranking candidates. Local audit: 72,374/4,717/5,465 train/valid/test facts,
237 relations, one component, no isolated entities. The old 1,000 pilot had 1,022
training facts; old and new MRR are not directly comparable across data scopes.
Do not silently vary sampling between configurations.

## Equations and implementation constraints

E is first-RGAT entity output; R is the ComplEx relation table; G2 is the second
RGAT. Residual: U=E+tanh(gE)*TE; final relations=R+tanh(gR)*TR;
final entities=U+tanh(gF)*G2(U,A). Raw scalar gates start at .01.
No-Refinement uses the same text fusion and omits G2/refinement gate.
Soft prompts project KG vectors to RoBERTa space, prepend one continuous token,
and project its contextualized output back to KG space. Independent text has no
KG-to-LM projection/structural input; use masked mean pooling of frozen outputs,
then a live trainable 768-to-32 projection. It retains G2 and residual refinement.

Do not change RGAT, fusion/refinement, pooling, projection, ComplEx or BCE math.
Freeze RoBERTa and keep it eval; allow gradients through its soft-prompt operations.
Cache only independent frozen pooled vectors, not projected or structural outputs.
Encode graph once per optimizer step; reuse for positives and negatives. Preserve
average rank for exact ties: 1+higher+(equal-1)/2. Optimistic MRR/ties are diagnostics.
Warmup validation evaluates the structural model and cannot select an integrated
checkpoint. Evaluate the initial integrated model at warmup transition (or epoch 0)
and allow that checkpoint to remain best. Label the two phases in plots/reports.

## Notebook protection and entry points

NEVER modify, overwrite, regenerate or revert notebooks/research_pipeline.ipynb.
The researcher manually edited it. Preserve all cells/outputs byte-for-byte.
New notebook: notebooks/warmup_ablation_5000.ipynb.
Separate generator: notebooks/build_warmup_ablation_notebook.py.
Historical generator can only write a separate non-overwriting preview.

- run_warmup_ablation.py: four configurations by three seeds, shared data and reports.
- run_phase5_comparisons.py: alias for the current twelve-run study.
- run_phase5_training.py: --configuration for the study or --architecture for a single model.
- run_research_baseline.py: independent full graph baseline, outside current comparison.
- run_subset_audit.py: real-data selection/density audit, no training.
- run_scratch_smoke_test.py: --offline-only --device cpu or --real-lm-only --device cuda.
- run_pretrained_check.py: mandatory notebook preflight; --inspect-file is read-only.
- run_resume_diagnostic.py: repeated exact controls/resume; optional actual RoBERTa.
- training/ablation_reports.py: phase-separated analysis, eight plots, paired comparisons.

## Reliability, resume and resource limits

Keep verified roberta-base revision e2da8e2f811d1448a5b465c236feacd80ffbac7b and
file checksums. In Colab set HF_HOME=/content/hf_cache and consistent Hub cache
paths before imports; remove stale TRANSFORMERS_CACHE. Copy verified backups
from Drive to local storage and validate before loading. Never clear a cache tree,
use random encoder weights, alternate models or pickle fallback to hide failures.
The unused AutoModel pooler is the only allowed missing pretrained component.

Preserve separate CPU tests, real-LM GPU tests and CUDA resume diagnostics with
streamed/saved logs. Corruption fixtures must not inherit RESEARCH_MODEL_BACKUP
from persistent Drive; isolate it in both the runner environment and test setup.
A failed mandatory pretrained preflight must block study launches even if optional
tests are disabled. Never report CPU checks as CUDA/T4 verification.

Resume only a run's own verified atomic last.pt: parameters/buffers, Adam state,
module modes, Python/NumPy/Torch CPU/all-CUDA RNG, negative sampler, shuffle generator,
progress/history and best selection. Restore and audit exactly before training.
Strict deterministic algorithms stay enabled; unsupported kernels fail explicitly.
No tolerance relaxation without GPU evidence. No scheduler/scaler exists under
fixed-LR/full-precision protocol. Preserve RNG schema and additive checkpoint
compatibility; old source/data identities require a new run directory.

Do not launch expensive twelve-run experiments during development. Use synthetic
fixtures and data-only audits. T4 feasibility of 5,000-entity integrated training
is unverified. If OOM occurs, report it; never silently change entities, width,
precision, objectives or optimizer-update budget. Text/eval chunk reductions need
consistent explicit settings/new tag and do not remove full-graph RGAT costs.
Preserve all old checkpoints/results and historical notebooks. Record environment,
source/LM/data identities, all phase budgets, timings, GPU peaks and diagnostics.
