# Scratch pipeline implementation and verification

This report describes the completed 1,000-entity stage. For the current twelve-run
5,000-entity warmup ablation, see [WARMUP_ABLATION_5000.md](WARMUP_ABLATION_5000.md).
The historical notebook/results remain preserved; active registrations have changed.

## Changed and added files

| Area | Files and purpose |
| --- | --- |
| Models | `models/frozen_lm.py`: independent masked-mean text encoding; `models/independent_text_bridge.py`: trainable projection without KG input; integration/refinement classes accept independent or soft-prompt text while retaining existing equations. |
| Factory | `training/variants.py`: five scratch architectures, graph-only baseline, and warmup-budget validation. |
| Training | `training/research_pipeline.py`: shared independent baseline / integrated trainer, structural warmup, validation selection, full test evaluation, atomic checkpoints, verified resume/skip. |
| Data/provenance | `training/experiment_io.py`: full-data validation, exact subset manifests, leakage checks, hashes, package/source metadata. |
| Text caching | `training/text_cache.py`: pooled frozen vectors only; token/model/pooling identities prevent unrelated cache reuse. |
| Scoring adapter | `evaluation/enriched_scorer.py`: the existing explicit-relation ComplEx candidate equations extracted for reuse. |
| CLI | `training/research_cli.py`, `run_research_baseline.py`, updated `run_phase5_training.py` and `run_phase5_comparisons.py`. |
| Comparisons | `training/comparisons.py`, `training/reports.py`: sequential independent runs, validation-only selection, CSV/JSON reports and PNG plots. |
| Compatibility | `training/legacy_phase5.py`: historical pretrained runner preserved behind explicit `--legacy-pretrained`. |
| Configuration | `experiments/configs/research_protocol.yaml`: selected 5+25 budget and all stage defaults. |
| Notebook | `notebooks/research_pipeline.ipynb` and its readable builder `notebooks/build_colab_notebook.py`. |
| Tests | `tests/`, `run_scratch_smoke_test.py`: offline protocol regression suite plus opt-in real-RoBERTa gate. |
| Documentation | README, current CLAUDE instructions, this report, and preserved historical notes. |

## Cleanup audit

**Deleted files: none.** Existing scripts remain referenced by historical
documentation, imports, checks, or checkpoint reproduction workflows; there was
no evidence justifying their deletion. No datasets, checkpoints, results, or user
files were removed. The notebooks/tests directories previously had no committed
notebooks/tests to replace.

The old Phase 5 runner was relocated into a named legacy module rather than
discarded. Its duplicate explicit-relation candidate scorer was extracted into
one shared implementation. The old 3-variant orchestration was replaced with the
5-variant workflow; the historical individual experiments remain reproducible via
the explicit legacy entry point. Generated checkpoints/caches/results are excluded
through `.gitignore`, not deleted from the user's workspace.

The unused `torchkge` install requirement was removed after checking code imports:
the repository implements its scorers locally. This does not change scoring math.
`nbformat` and `ipython` were added for notebook validation and execution checks.

## Validation scope

The 2026-10-10 model-loading/resume follow-up is documented in
[RELIABILITY.md](RELIABILITY.md), including separate root-cause findings, new
cache strategy, complete RNG restoration, strict deterministic execution,
independent Colab GPU gates and current CPU/GPU verification limits. The earlier
verification record below describes the original restructuring, not CUDA approval.

Local verification on 2026-10-09: **10 regression tests passed**, including
notebook schema/ordered execution and all fifteen orchestration configurations.
Python syntax checks passed for 59 source files. Existing ComplEx and exact-tie
known-answer checks passed. The real pretrained RoBERTa gate passed for all five
variants on CPU. Runtime packages: PyTorch 2.11.0+cpu, Transformers 5.13.0,
PyTorch Geometric 2.8.0.post1. These are observed test versions, not an assertion
that other untested combinations work identically. No GPU benchmark was run.

Regression checks cover:

- Real RGAT/ComplEx with every scratch architecture, finite forward/backward paths,
  parameter gradients, unchanged frozen-LM weights, and checkpoint round trips.
- Exact original/no-warmup parameter equality before training and explicit budget
  accounting; the actual training objectives differ only during warmup epochs.
- No structural inputs/soft prompts in independent text encoding, masked pooling,
  cache hits/invalidation, live projection gradients, residual gates and refinement.
- Original split provenance and exact training-only message edges.
- Tiny independent baseline and variant training runs, output artifacts,
  incompatible-run rejection, and interrupted/resumed CPU training equivalence.
- Complete variant/seed aggregation, sample SD, validation-only selection even when
  test rankings disagree, CSV exports and plot generation.
- Notebook schema, code syntax and ordered execution with external/training actions
  replaced by fixtures (not a real Colab research run).
- Existing ComplEx formula and known-answer exact-tie ranking scripts.

The optional `python run_scratch_smoke_test.py --real-lm` uses actual pretrained
RoBERTa and a tiny synthetic graph. **All five variants passed this gate locally
on CPU using the existing cached pretrained weights**, including forward/backward,
component gradients and unchanged frozen-LM weights. Rerun it in the intended Colab
environment before expensive experiments. The broader offline regression suite uses
a deterministic LM substitute. No CUDA or full-data performance is claimed. Full
100-epoch/15-run research experiments are not part of implementation validation.

## Explicit remaining limitations

- Full-data integrated training has not been demonstrated to fit a T4; graph
  activations can dominate memory even with LM checkpointing and text chunks.
- Hugging Face and dataset downloads need network access the first time. Keep the
  saved resources/versions stable; source, data, text and environment identities are
  recorded. Exact reproducibility across GPU scatter implementations is not promised.
- Default final training uses one seed and a different updates-per-epoch budget from
  the graph baseline. Reports expose those differences; stronger equal-compute and
  multi-seed final comparisons require additional experiments.
- Historical pretrained results are a different protocol. This implementation does
  not convert them into scratch results or claim any new performance improvement.
