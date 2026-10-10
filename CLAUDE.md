# CLAUDE.md — current project instructions

Read README.md and experiments/configs/research_protocol.yaml for the current
workflow. Historical research notes and recorded Colab outputs are preserved in
docs/HISTORICAL_NOTES.md. That archive contains superseded architecture decisions
and optimistic-tie metrics; it is not the current implementation plan.

## Research task and architecture

KG link prediction on FB15k-237, combining graph structure with textual semantics.
Current structural model: two-layer RGAT, one head, width 32, ComplEx scoring.
The graph/text models use frozen RoBERTa and a second two-layer RGAT where specified.
Entities are graph-encoded; relation vectors come from the learned ComplEx table.
There is no formal ontology/OWL reasoner in the implemented scope.

## Authoritative experimental protocol

1. Independently train a graph-only RGAT + ComplEx baseline on all 272,115
   FB15k-237 training facts for 100 epochs, validation every 5 epochs, and report
   full validation/test metrics. Its checkpoint is for evaluation/comparison only.
2. Train five variants FROM SCRATCH on one fixed seed-0 1,000-entity subset with
   training seeds 0/1/2 (15 independent runs): original, residual, no-refinement,
   original-no-warmup, residual-no-softprompt.
3. User-selected total budget is 30 epochs: 5 structural warmup + 25 integrated
   epochs for four variants; 0 + 30 for original-no-warmup. Pilot training has one
   optimizer step per epoch. Do not silently add warmup epochs to this budget.
4. Choose the architecture using mean VALIDATION MRR, never test MRR. Report
   sample standard deviation, test metrics, training losses, duration and memory.
5. Optional final full-data run starts a NEW model from scratch. Never initialize
   from the baseline or pilot checkpoint. Notebook flag defaults to disabled.

Scratch applies to all task-specific parameters. Pretrained frozen RoBERTa is
intentional. Resume is permitted only from the SAME run's verified last.pt.

## Warmup definition

The previous workflow used a full-dataset-pretrained baseline as implicit warmup;
there was no separate structural-warmup loop or LR scheduler. The new explicit
warmup optimizes graph-only BCE from scratch, then continues with the integrated
objective at the same fixed learning rate, retaining structural Adam moments.
All modules are initialized before warmup. Original/no-warmup have identical
architecture and initial task parameters for a paired seed. Warmup structural
validation cannot select an integrated-model checkpoint. The first integrated
validation is at the transition before any integrated update and can remain best.

## Equations to preserve

B denotes the shared KG-conditioned soft-prompt bridge, G2 the second RGAT.
Original: TE=B(E,text), TR=B(R,text), final entities=G2(TE,A), final relations=TR.
Residual: U=E+tanh(gE)*TE; final relations=R+tanh(gR)*TR;
final entities=U+tanh(gF)*G2(U,A).
No-refinement: same gated text fusion; omit G2 and its gate.
Original-no-warmup: exactly Original, with structural warmup disabled.
Residual-no-softprompt: same Residual fusion AND G2, but TE/TR come from independent
frozen text encoding, attention-mask mean pooling, and trainable 768-to-32 projection.
No structural token or KG-to-LM projection exists in that variant.
Scalar gates retain initial raw value 0.01 and tanh bounding.

## Non-negotiable implementation rules

- Do not load trained KG baseline parameters into any scratch variant, directly
  or through helpers/cached graph vectors. Factory accepts no checkpoint argument.
- Build message-passing edges from training facts only. Preserve original splits.
  Use held-out labels only in evaluation and filtered evaluation candidate removal.
- Reject sampled negatives matching any original training fact within the candidate
  subset, even if omitted by the connected graph sampler. Do not use held-out
  labels to filter training negatives.
- Keep BCE/logistic loss and the existing ComplEx mathematics unchanged.
- Use average ranks for exact ties: 1+higher+(equal-1)/2. Optimistic_MRR and tie
  diagnostics are additional reports, not selection metrics.
- Freeze RoBERTa weights and keep it in eval mode. Soft-prompt paths need gradients
  through LM operations. Independent text encoding may use no_grad and cache only
  frozen pooled vectors; never cache projected/trainable or KG-conditioned outputs.
- Encode the graph once per optimizer step; reuse vectors for sampled positives
  and negatives. Keep the learning rate fixed throughout a run.
- Record dataset/subset/token/code/environment identities, epochs by phase,
  optimizer steps, configurations, seeds, best checkpoint and results.
- Do not equate subset MRR with full-dataset MRR or equal epochs with equal compute.
- Do not invent performance results or claim that residuals, text, or warmup help
  before controlled corrected-evaluator experiments establish that.

## Main entry points

- notebooks/research_pipeline.ipynb: complete 16-section Colab workflow.
- run_research_baseline.py: independent full baseline with persistent artifacts.
- run_phase5_training.py: scratch variants, --max-entities 0 for full dataset.
- run_phase5_comparisons.py: five variants by three seeds, shared subset, reports.
- run_scratch_smoke_test.py: --offline-only --device cpu; --real-lm-only --device cuda.
- run_pretrained_check.py: mandatory notebook model-load/forward preflight; read-only --inspect-file diagnosis.
- run_resume_diagnostic.py: repeated exact control/resume comparisons; --real-lm tests actual RoBERTa.
- run_complex_scorer_check.py and run_tie_ranking_check.py: mathematical checks.
- training/research_pipeline.py: shared training, warmup, resume and test evaluation.
- training/reports.py: validation-only selection, exports, figures, final comparison.

Old Phase 0–4 gates and baseline trainers remain useful. Historical Phase 5 runs
are accessible ONLY via explicit --legacy-pretrained and training/legacy_phase5.py.
The new notebook/comparison runner never uses that mode. Preserve historical
checkpoint reevaluation and do not mix its results with scratch experiments.

## Validation and resource limits

Use lightweight offline regression tests and the real-LM gate where feasible.
Do not run expensive research experiments merely to validate code changes.
Offline tests use real RGAT/ComplEx and a small LM substitute, not actual RoBERTa.
Full integrated GPU feasibility is not established; text batching does not solve
all full-graph RGAT allocations. Do not silently reduce graph/dimensions, alter
precision or change objectives after OOM. No full LM fine-tuning is requested.

Store checkpoints directly on Drive, use fresh output directories for changed
protocols, and preserve completed results. --skip-completed verifies compatibility;
--resume restores the same run's parameters/buffers, Adam state, module modes,
Python/NumPy/Torch CPU/all-CUDA RNG streams, negative sampler and shuffle generator.
Strict deterministic algorithms are enabled, including PyTorch CUDA scatter where
supported. Unsupported deterministic kernels must fail, never silently fall back
to nondeterministic training. Exact resume tests are retained; do not relax their
tolerances without evidence from GPU controls. Fixed LR/full precision means no LR
scheduler or gradient scaler state. Save only epoch-boundary resumable checkpoints.
Resume audits must pass before another optimizer update. Numerical policy is part
of run identity. New RNG schema/source identity requires a fresh output directory;
do not fabricate missing RNG states to migrate old experiments.

## Verified pretrained model loading

The Hub model remains roberta-base, pinned to
e2da8e2f811d1448a5b465c236feacd80ffbac7b. models/pretrained.py validates safetensors
headers/layouts and pinned file hashes, stages files locally, and uses explicit
local_files_only/use_safetensors loading. Never use random weights, another model,
or a pickle-format fallback to hide a load failure. An unused AutoModel pooler is
the only allowed missing component; all encoder weights must load.

In Colab set HF_HOME=/content/hf_cache and HF_HUB_CACHE=/content/hf_cache/hub before
imports. Remove stale TRANSFORMERS_CACHE. RESEARCH_MODEL_BACKUP points to a verified
Drive backup; weights are copied to local storage and checked before loading.
Persistent checkpoints/results/pooled text remain on Drive. Repair only confirmed
bad files; never clear the entire HF cache or touch unrelated artifacts. Explicit
local model directories are read-only and require safetensors.

Notebook preflight must pass before baseline, pilot or final launches, regardless
of optional test flags. CPU offline tests, GPU real-model tests, and dedicated GPU
resume diagnostics are separate groups with persistent logs. See
docs/RELIABILITY.md for diagnosis, commands and actual verification limits.

## Cleanup discipline

Trace imports and documentation references before removing files. Existing models,
scorers, dataset utilities, useful gates, checkpoints and historical outputs remain
valuable. No files were deleted during this restructure. The former Phase 5 trainer
was retained in a legacy module and its duplicate candidate scorer was shared.
Generated data/caches/checkpoints/results are ignored by Git, not automatically
deleted. See docs/IMPLEMENTATION.md for the file-change and validation report.
