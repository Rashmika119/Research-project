# Three-pipeline repository cleanup

Completed source cleanup on `cleanup/three-selected-pipelines`, based on
`d9348c744da7c30c47e585de9e9ed4674a889399` (pre-cleanup `feature/phase_03`).
The initial working tree was clean. Changes remain uncommitted for review; no
push, history rewrite, checkpoint migration, or experiment training was performed.

The approved default remains full FB15k-237, 100 total epochs, 10 structural
warmup epochs when enabled, seed 0, and exactly three independent runs. Each
run has 900 optimizer updates. Architecture equations and optimization are unchanged.

## Audit and dependency findings

[CLEANUP_AUDIT.json](CLEANUP_AUDIT.json) records the pre-cleanup tracked tree,
file hashes, static imports, proposed KEEP/MODIFY/REMOVE/ARCHIVE classifications,
final classifications, and protected evidence hashes. Runtime imports, local imports
inside functions, string architecture registrations, notebook commands and test
dependencies were also inspected. No unresolved dynamic architecture import was found.

The selected dependency path is:

```text
run_full_dataset_comparison -> training.full_study -> training.research_pipeline
  -> variants -> model_factory -> KGOnlyBaselineRGAT -> RGATEncoder + ComplExScorer
  -> KGTextRefinement -> KGTextIntegration (shared module layout only)
       -> IndependentTextBridge -> FrozenLM + LMKGProjection
       -> second RGATEncoder + unchanged gated residual equations
  -> experiment_io / reproducibility / text_cache / loss / sampling / evaluation
  -> full_reports -> reports (shared CSV and descriptive statistics)
```

## Deleted source files (29)

All were confirmed outside the retained dependency graph after shared-helper
extraction. Historical notebook references intentionally resolve in their matching
old source revision. No deleted source file contains unique saved experiment evidence.

| File | Reason |
| --- | --- |
| `models/kg_encoder.py` | Unused RGCN encoder; selected structural/refinement modules use RGATEncoder. |
| `models/kg_lm_bridge.py` | KG-conditioned soft-prompt bridge; no selected caller after integration-base simplification. |
| `models/kg_lm_projection.py` | KG-to-LM prompt projection; independent text retains the separate LM-to-KG projection. |
| `models/kg_only_baseline.py` | RGCN baseline; selected baseline is KGOnlyBaselineRGAT. |
| `notebooks/build_colab_notebook.py` | Obsolete generator; historical notebook is protected evidence and must not be regenerated. |
| `notebooks/build_warmup_ablation_notebook.py` | Obsolete generator; historical notebook is protected evidence and must not be regenerated. |
| `preprocessing/toy_subset.py` | Legacy non-induced toy subset; active full-data loading and shared induced-subset checks do not import it. |
| `run_phase0_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase1_full_training.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase1_rgat_full_training.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase1_rgat_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase1_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase2_real_text_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase2_relation_text_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase2_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase3_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase4_smoke_test.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase5_comparisons.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_phase5_training.py` | Historical phase-specific launcher/smoke check; only retired workflows import it. Preserved in the pre-cleanup Git revision. |
| `run_reevaluate_checkpoint.py` | Phase 1/5 legacy checkpoint reevaluator importing removed runners; recover matching historical source to reevaluate those artifacts. |
| `run_research_baseline.py` | Duplicate baseline launcher; main runner supports --single baseline. |
| `run_subset_audit.py` | Standalone comparison with the retired toy-subset algorithm; saved subset audit remains unchanged. |
| `run_warmup_ablation.py` | Retired four-configuration pilot launcher. |
| `tests/test_warmup_ablation.py` | Four-way/soft-prompt tests retired; relevant subset checks retained in tests/test_dataset.py. |
| `training/ablation_reports.py` | Retired four-way reporting; METRICS, mean_sd and loss_behavior moved verbatim into training/reports.py. |
| `training/comparisons.py` | Pilot/four-way study orchestration replaced by the full-study runner. |
| `training/legacy_phase5.py` | Pretrained-baseline Phase 5 training; selected workflows always start from scratch. |
| `training/research_cli.py` | Obsolete single-architecture CLI exposing retired experimental options. |
| `training/train_kg_baseline.py` | Legacy configurable RGCN/RGAT baseline trainer; full study owns selected baseline training. |

## Archived configurations (9)

Each file moved byte-for-byte from `experiments/configs/` to `docs/archive/configs/`.
Git retains the originals in the recorded revision. Archive hashes were checked.

- [full_dataset_300.yaml](archive/configs/full_dataset_300.yaml)
- [phase0_fb15k237.yaml](archive/configs/phase0_fb15k237.yaml)
- [phase1_full.yaml](archive/configs/phase1_full.yaml)
- [phase1_rgat_full.yaml](archive/configs/phase1_rgat_full.yaml)
- [phase1_rgat_toy.yaml](archive/configs/phase1_rgat_toy.yaml)
- [phase1_toy.yaml](archive/configs/phase1_toy.yaml)
- [phase2_lm_toy.yaml](archive/configs/phase2_lm_toy.yaml)
- [research_protocol.yaml](archive/configs/research_protocol.yaml)
- [research_protocol_1000_historical.yaml](archive/configs/research_protocol_1000_historical.yaml)

## Modified files

| File | Change |
| --- | --- |
| `CLAUDE.md` | Selected-only scope, notebook protection, exact-resume policy and no-training verification instructions. |
| `README.md` | Only the three selected training configurations, launch instructions and cleanup/recovery links. |
| `docs/FULL_DATASET_300.md` | Active protocol remains 100/10; document archived YAML, historical resume and future independent seeds. |
| `models/__init__.py` | Update stale historical module documentation only; numerical code unchanged. |
| `models/frozen_lm.py` | Remove prompt injection API; independent masked mean pooling, frozen weights and eval mode remain unchanged. |
| `models/kg_encoder_rgat.py` | Update stale historical module documentation only; numerical code unchanged. |
| `models/kg_only_baseline_rgat.py` | ComplEx default and current docstring; selected construction is unchanged. |
| `models/kg_text_integration.py` | Keep the minimal shared base, module names and registration order; remove soft-prompt bridge and retired forward path. |
| `models/kg_text_refinement.py` | Remove Original/No-Refinement/prompt branches; preserve selected gates, refiner construction and arithmetic. |
| `models/scorer.py` | Remove DistMult class/registration; ComplEx formulas unchanged. |
| `preprocessing/__init__.py` | Update stale historical module documentation only; numerical code unchanged. |
| `requirements.txt` | Comments updated; no dependency removed because all packages serve retained code or verification. |
| `run_resume_diagnostic.py` | Opt-in diagnostic accepts only the retained text architecture. |
| `run_scratch_smoke_test.py` | Remove retired prompt assertions; retain opt-in actual-RoBERTa verification for the selected text model. |
| `tests/test_full_study.py` | Require selected-only registry; remove obsolete generator import; check allowed future independent seeds. |
| `tests/test_notebook.py` | Replace pilot execution/orchestration tests with exact historical notebook preservation checks. |
| `tests/test_research_pipeline.py` | Remove retired-only report assertions; selected architecture checks and opt-in training/resume regressions remain. |
| `training/full_reports.py` | Import shared helpers from retained module; explicitly support another independent single seed, default 0. |
| `training/full_study.py` | Default three seed-0 runs unchanged; explicit future --seed N propagates to child configs and reports. |
| `training/losses.py` | Update stale historical module documentation only; numerical code unchanged. |
| `training/model_factory.py` | Remove RGCN construction; keep shared RGAT factory and all selected settings. |
| `training/reports.py` | Retain CSV/run exports and shared statistics; remove pilot selection/plot pathways. |
| `training/research_pipeline.py` | Change only the retired default architecture name to residual-no-softprompt; training/resume body unchanged. |
| `training/variants.py` | Only selected baseline/text configurations; one shared text model for both schedules. |

New verification/documentation files: `run_static_checks.py`,
`tests/test_selected_models.py`, `tests/test_dataset.py`,
`experiments/configs/README.md`, this report, the audit JSON, and
`docs/archive/README.md`. The main notebook and its generator were not changed.

## Shared dependencies deliberately retained

- `KGTextIntegration`: minimal shared initializer preserves class/module layout and initialization order; it is not a separate trainable architecture.
- `KGTextRefinement`, `KGOnlyBaselineRGAT`, `RGATEncoder`, `ComplExScorer`: the two actual implementations and shared graph/scoring mathematics.
- `IndependentTextBridge`, `FrozenLM`, `LMKGProjection`, `pretrained.py`, `text_cache.py`: frozen loading, masked pooling, caching and live trainable projection.
- `experiment_io.py`, `reproducibility.py`, `research_pipeline.py`: atomic checkpoints, RNG state, identities, optimizer and validation selection. Historical checkpoint metadata fields remain intact.
- `preprocessing/induced_subset.py`: shared graph statistics, provenance and offline diagnostic data handling. The active runner always requests the complete dataset.
- Dataset/text loading, graph builder, BCE loss, negative sampling and evaluation modules remain in place.
- Text alignment, ComplEx, tie-ranking, pretrained and opt-in resume/smoke verification scripts remain useful to selected workflows.

## Remaining entrypoints and configurations

The only research-training entrypoint is `run_full_dataset_comparison.py`:

- `baseline`: first RGAT (2 layers) and ComplEx, no LM, no warmup.
- `residual-no-softprompt-warmup`: first and second RGAT (2 layers each), frozen independent text, 10 warmup + 90 integrated epochs.
- `residual-no-softprompt-no-warmup`: the identical model, 100 integrated epochs.

The only active YAML is `experiments/configs/full_dataset_100.yaml`, unchanged.
The active notebook remains `notebooks/full_dataset_comparison_300.ipynb`,
unchanged including all cells/output records. Its `_300` name is historical.
`run_static_checks.py` never trains. `run_scratch_smoke_test.py` and
`run_resume_diagnostic.py` remain optional developer diagnostics because the
active notebook calls them only behind flags that remain false. They are not
additional research workflows. Other retained `run_*_check.py` scripts verify
pretrained files, text alignment or scoring.

## Verification and results

- `python run_static_checks.py`: **24 tests passed** on CPU, no optimizer steps. Covers model layers, scratch pairing, cache reuse, projection/graph/gate derivatives, checkpoint and Adam/RNG restoration, identity rejection, selection, reports/13 plots, mocked notebook execution, corruption handling and protected notebooks.
- `python run_complex_scorer_check.py`: passed algebra, candidate scoring and derivative checks.
- `python run_tie_ranking_check.py`: passed known-answer filtered ranking and exact-tie checks.
- `python run_full_dataset_comparison.py --help`: only the three selected configuration choices are exposed.
- AST/import audit: all remaining production modules import; local imports resolve; 52 Python source files parsed (before adding the configuration README, which contains no Python).
- `git diff --check`: passed.
- Local pre/post verifier `python experiments/research/cleanup_equivalence.py capture` (before edits) / `compare` (after edits): initialized full state dictionaries, parameter order, trainable flags, RNG after initialization, eval/dropout-mode forwards and all parameter gradients are **exactly equal** for baseline and No-Softprompt on CPU. Uses deterministic LM fixtures, zero optimizer steps. Reference tensor snapshot and script remain ignored local verification artifacts.
- Real full-data audit via `load_dataset(download_if_missing=False, use_synthetic_fallback=False)`, `prepare_data(..., max_entities=0)` and `verify_manifest`: passed 14541/237 entities/relations, 272115/17535/20466 facts, 544230 message edges, exact train-only graph/leakage checks, 36 isolated entities and 41 components. No download or training occurred.
- Dataset SHA256: `e465b6f9eee2bc96d507ed6903116fe73b920e5f35163dc7d812455c10b70dfe`.
- Manifest SHA256: `64eb297df2f1fc762044aef49efd3b6aabe781e16cab41fdce5fccfa70d34ed3`.
- Preservation: 239 accessible data/evidence/notebook files rehashed unchanged, including all three notebooks; three restricted raw split files separately read and hashed during the approved full-data audit. All nine archived YAMLs match their pre-cleanup hashes. No Drive directory was accessed or deleted.

CPU logs: `experiments/research/cleanup_final_checks.log`; report fixture numbers
are fabricated for validation and are never represented as experimental results.

## Checkpoint compatibility and limitations

Selected state-dict names/shapes, parameter registration order and initialization
are preserved, with exact CPU pre/post evidence as above. Checkpoint payload and
restore schemas, source/environment checks, and exact RNG/Adam audits were not
weakened. Non-training checkpoint tests use fabricated nonzero Adam moments and
weights to exercise save/load/restoration without an optimizer step.

Cleanup changes the source fingerprint. **Resume an existing experiment with its
exact recorded source revision and packages**, not this cleanup revision. The
pre-cleanup commit preserves all removed code; earlier runs may require earlier
commits. Historical full-object checkpoints were not migrated. Retired models
are loadable using their matching historical code, not through the active registry.

Completed full-dataset selected-model results/checkpoints were not present in this
development checkout. Only earlier 12-entity diagnostic results were found.
Regeneration from actual completed full-study results, actual selected research
checkpoint loading, GPU resume/training, and actual RoBERTa forward/backward
are therefore **unverified in this cleanup**. No GPU, synthetic training, smoke
or one-epoch training was started. Frozen-LM derivative tests use explicit fixtures.
History documents and the two protected historical notebooks stay in place to
preserve links and evidence. Their retired commands apply only at old revisions.

## Launch the three selected pipelines

Use a fresh output path after committing/pushing this source; in Colab set the
notebook REPO_REF to that exact pushed commit and choose a fresh RUN_TAG.

```bash
python run_full_dataset_comparison.py --output-dir experiments/research/full_100_seed0_cleanup_run1 --skip-completed --resume
```

This defaults to seed 0 and three runs. A future independent seed uses
`--seed N` with a separate output path; pass the same seed for report-only or
resume. Budgets and architectures stay fixed. Reports are single-seed; combined
multi-seed aggregation is outside this cleanup.

For historical source recovery and preserved documents see
[the archive index](archive/README.md).
