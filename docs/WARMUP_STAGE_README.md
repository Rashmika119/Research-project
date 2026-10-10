# Structural and textual knowledge-graph learning

Research code for filtered head/tail link prediction on FB15k-237, combining RGAT
structure and frozen pretrained RoBERTa descriptions. The implemented scope is
relational triples, not a formal OWL reasoner.

## Current study: 5,000-entity warmup ablation

Open [notebooks/warmup_ablation_5000.ipynb](notebooks/warmup_ablation_5000.ipynb),
select a T4 runtime, choose a pushed repository ref containing this stage, and run
its eight sections in order. This launches twelve independent runs by default.
The user's manually edited [previous notebook](notebooks/research_pipeline.ipynb)
is preserved unchanged and is a historical artifact.

| Configuration | Architecture | Structural epochs | Integrated epochs |
| --- | --- | ---: | ---: |
| residual-warmup | Residual, soft prompts | 5 | 25 |
| residual-no-warmup | Same Residual model | 0 | 30 |
| residual-no-softprompt-warmup | Residual, independently pooled text | 5 | 25 |
| residual-no-softprompt-no-warmup | Same independent-text model | 0 | 30 |

Every configuration uses seeds 0/1/2, the same exact 5,000 entities, 30 optimizer
updates, width 32, Adam LR .001, four negatives, BCE and corrected tie-aware filtered
evaluation. All trainable KG components start from scratch. Only frozen RoBERTa
weights and independent frozen text encodings may be reused. Warmup retains the
same run's structural parameters/Adam moments; no weights move between runs.
Mean validation MRR selects the configuration, never test scores.

Available architectures are `residual`, `residual-no-softprompt`, and
`no-refinement`. No-Refinement is retained for future research, outside this study.
The independent full-data graph baseline remains available. Original and Original
No-Warmup are no longer active factory/CLI options. Shared historical model paths
and explicit legacy reevaluation remain for reproducibility.

## Data and unchanged model mathematics

The new train-only BFS selects entities deterministically, then retains all induced
training facts. Local seed-0 audit: **5,000 entities, 72,374 training facts, 4,717
validation and 5,465 test facts**; one connected training component, no isolated
entities. The old traversal kept 7,221 training facts for exactly the same selected
entities. Original split boundaries and relation IDs remain intact. All twelve
runs verify one saved manifest and rank against the same 5,000 candidates.

Residual equations remain `U = E + tanh(gE)*TE`,
`final_entities = U + tanh(gF)*G2(U,A)`, and
`final_relations = R + tanh(gR)*TR`; raw gates start at .01. Soft prompts condition
frozen RoBERTa on structure. Independent text uses masked mean pooling and a live
trainable 768-to-32 projection, with the same second RGAT/residual refinement.
No-Refinement keeps fusion but omits G2. ComplEx scoring, BCE, negative sampling,
RGAT equations and average ranks for exact ties are unchanged.

Only frozen pooled language vectors may be cached, never trainable projections or
KG-conditioned text outputs. RoBERTa remains frozen/eval while soft-prompt gradients
flow through it. The verified pinned loader, targeted corrupt-file recovery,
local Colab model storage and verified Drive backup remain in use.

## Commands

Run from the repository root after `pip install -r requirements.txt`.

```bash
# Data audit only: no model training
python run_subset_audit.py

# Independent reliability gates
python run_scratch_smoke_test.py --offline-only --device cpu
python run_pretrained_check.py --device cuda
python run_scratch_smoke_test.py --real-lm-only --device cuda
python run_resume_diagnostic.py --device cuda --repeats 3 --output-dir experiments/research/cuda_check

# Twelve runs, safe resume and completed-run verification
python run_warmup_ablation.py --max-entities 5000 --epochs 30 --warmup-epochs 5 --seeds 0 1 2 --output-dir experiments/research/warmup5000_run1 --skip-completed --resume

# One configuration, independently from scratch
python run_phase5_training.py --configuration residual-no-warmup --output-dir experiments/research/single

# Retained architecture, outside the ablation
python run_phase5_training.py --architecture no-refinement --warmup-epochs 5 --output-dir experiments/research/separate_no_refinement

# Independent full graph-only baseline, outside the ablation
python run_research_baseline.py --output-dir experiments/research/baseline
```

The old `run_phase5_comparisons.py` entry point now invokes the same twelve-run
study. [research_protocol.yaml](experiments/configs/research_protocol.yaml)
documents defaults. CLI flags and explicit notebook settings control execution;
YAML edits alone do not override CLI defaults. Changed data/code/packages/settings
require a new output directory. Existing checkpoints/results are never deleted.

## Colab output and results

The new notebook prints configuration/architecture/warmup/seed and each completed
epoch's phase, loss, validation MRR, validation Hits@10, peak tracked GPU memory
and elapsed time. CPU regressions, real-LM GPU tests and CUDA resume diagnostics
have independent cells. Pretrained preflight must pass before the launch.

Each run saves config/manifest, best/last checkpoints with optimizer/RNG state,
history JSON/CSV, metrics, final results and timing. A partial run resumes only its
own atomic epoch-end checkpoint after exact restoration checks. Warmup validation
is structural diagnostic output; integrated validation selects checkpoints.

The final analysis produces mean/sample-SD tables, per-seed paired ablations,
loss/convergence/spike/plateau diagnostics, corrected versus optimistic MRR/tie
warnings, runtime comparisons and eight PNG plots. Warmup and integrated curves
are distinguished, never silently pooled. Full artifacts and logs go to the new
Drive stage directory, separate from the completed 1,000-entity stage.

See [the complete study guide](docs/WARMUP_ABLATION_5000.md) for all settings,
commands, statistical cautions, outputs, data audit and resource limitations.
Three seeds support descriptive comparisons, not unsupported significance claims.
The full 5,000-entity training experiment and T4 capacity remain unverified locally.
There is no automatic entity/width/precision/objective reduction after OOM.

## Historical provenance

- [User-reported stage-1 results](docs/STAGE1_RESULTS.md).
- [Preserved stage-1 protocol](docs/STAGE1_PROTOCOL.md) and
  [historical YAML](experiments/configs/research_protocol_1000_historical.yaml).
- [Original implementation report](docs/IMPLEMENTATION.md).
- [Model-loading and resume reliability investigation](docs/RELIABILITY.md).
- [Older research notes](docs/HISTORICAL_NOTES.md).

Earlier-stage scores use a different entity scope and subgraph strategy; do not
compare them directly as evidence of architectural improvement. Reproduce old
notebooks/checkpoints with their recorded code revision. Never run the historical
notebook generator to update the new study: use
`python notebooks/build_warmup_ablation_notebook.py` for the new notebook only.
