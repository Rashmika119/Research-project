# Full FB15k-237 three-model study

The researcher approved this protocol after source inspection. Open
`notebooks/full_dataset_comparison_300.ipynb`; the two earlier notebooks are
protected historical artifacts, including their manual edits and saved outputs.
This is a new study. Earlier 100-epoch baseline scores are not equivalent controls.

Updated at the researcher's request: **100 total epochs, 10 warmup epochs, seed 0
only (three experiments)**. The notebook and guide retain their `_300` filenames
for existing links; their active settings are now 100/10. The new run tag is
`full_fb15k237_100_seed0_v1_run1` under `Research/full_dataset_comparison_100`.
Old 300-epoch artifacts remain untouched and cannot resume under these changed
settings. Use the old source revision to finish an old run. The old 300-epoch
YAML is archived at `docs/archive/configs/full_dataset_300.yaml`; the only active
YAML is `experiments/configs/full_dataset_100.yaml`.

## Fixed protocol

| Configuration | Warmup epochs/updates | Main epochs/updates |
| --- | ---: | ---: |
| baseline | 0 / 0 | 100 / 900 baseline |
| residual-no-softprompt-warmup | 10 / 90 | 90 / 810 integrated |
| residual-no-softprompt-no-warmup | 0 / 0 | 100 / 900 integrated |

Seed 0 only per configuration; three runs. Each run initializes all trainable
parameters from scratch. Same-seed No-Softprompt pairs have matching complete
trainable fingerprints; all three share their initial structural fingerprint.
No transfer from another configuration, seed, baseline or historical checkpoint.
The same run's Adam moments and structural parameters continue through warmup.

- Full FB15k-237 (`max_entities=0`): 14,541 entities, 237 scoring relations,
  272,115 train / 17,535 valid / 20,466 test facts. All entities are candidates.
- Original mappings/splits; train-only graph plus inverses: 544,230 message edges
  and 474 message relation types. No held-out message edges. Negative filtering
  uses known training facts; evaluation filtering uses all splits.
- 100 epochs, 32,768 positive triples per optimizer batch: eight full batches
  and one 9,971-positive batch. No drop-last, accumulation or early stopping.
  Exactly nine updates/epoch, 900 committed updates/run, 2,700 across the study.
- Adam LR .001, weight decay .00001, gradient clipping 1, four negatives/positive,
  existing positive BCE + negative BCE; no scheduler or gradient scaler.
- Full precision; width 32 (16 real + 16 imaginary ComplEx coordinates),
  two-layer/one-head RGATs, dropout .2; projection dropout .1.
- Frozen verified RoBERTa-base revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`; text batch 256, length 64;
  evaluation batch 512. Text/eval chunk changes require explicit settings and a
  new consistent run tag; the CLI never adapts them after OOM.
- Strict deterministic algorithms; Python/NumPy/Torch CPU/all-CUDA RNG,
  separate negative-sampling RNG and shuffle generator retained.

`experiments/configs/full_dataset_100.yaml` documents this protocol;
`training/full_study.py` constructs and validates it. YAML alone does not override
the runner. Fixed epochs/budgets cannot be changed accidentally by CLI flags.
The active registry contains only the three selected configurations. Unselected
architecture implementations and historical training entrypoints were removed.
See [the cleanup report](CLEANUP.md) for dependency and compatibility checks.

## Mathematics and caching

Baseline: first RGAT -> ComplEx; no text downloads or LM construction.
No-Softprompt: independent frozen masked-mean RoBERTa vectors -> trainable
768-to-32 projection; U=E+tanh(gE)*TE; final entities=U+tanh(gF)*G2(U,A);
final relations=R+tanh(gR)*TR. Raw gates start at .01. No soft prompt is injected.
Graph/fusion/refinement/scorer/loss equations and sampling are unchanged.

Cache only frozen pooled vectors (including unmasked special tokens). The
projection remains live and trainable. The content-addressed cache includes
tokens/order, model revision, runtime/device, dtype and batch size. Full-data
cache keys differ from the 5,000-entity pilot. Cache files persist on Drive.
RoBERTa is instantiated on CPU once per text child to preserve model construction
and initialization behavior, and moved to GPU only for missing pooled features.
It remains on CPU on cache hits. It is not reloaded or evaluated per optimizer
step. Trainable pooled projections and graph embeddings are never reused across
updates. Graph encoding runs once per optimizer step for positives and negatives.

Weights load from local `/content/hf_cache/hub`; verified persistent backups
remain under the prior shared resource directory. The new notebook's file-only
preflight uses `prepare_pretrained`, without instantiating a model or running
a forward/backward. Actual text training retains the existing strict loader.

## Validation and selection

Validation runs at epochs 5,10,...,100. Baseline and no-warmup eligible checkpoints
start at epoch 5. Warmup diagnostics at 5..10 are structural-only; the initial
integrated boundary evaluation after epoch 10 is diagnostic-only. Its first
eligible integrated checkpoint is epoch 15. Initial no-warmup epoch-0 evaluation
is also diagnostic-only. The result records this evaluation separately with its
boundary epoch. It is committed with the first main epoch, so a disconnection
before that commit can repeat the boundary evaluation without changing learning.

`selection_policy=scheduled_only` implements this approved change; historical
checkpoints may record `initial_and_scheduled`; the shared checkpoint schema
retains that value for explicit diagnostics. Equal validation MRR keeps the earliest
eligible checkpoint. After all 100 epochs, reload the selected model, reevaluate
validation and evaluate test. Test never selects checkpoints/configurations.
Select configuration by mean selected validation MRR for seed 0; exact
ties use configured order. Report seed-0 differences; across-seed SD is null and error bars are omitted,
without significance claims. Corrected ranking uses 1+higher+(equal-1)/2.

## Launch and resume

1. Push the updated code. Open the **new** notebook in Colab and select T4 GPU.
2. In Section 2 set `REPO_REF` to the exact pushed commit and choose a unique
   `RUN_TAG`. This notebook verifies checkout even if the Colab clone exists.
3. Run Sections 1–4. They mount Drive, install/record dependencies, validate
   dataset counts/manifest and pretrained file integrity. No training occurs.
4. Leave all four Section 5 optional training flags false.
5. Execute Section 6 when ready to start the real three-run study.
6. Inspect Section 7 checkpoints and Section 8 complete reports.

Default output: `MyDrive/Research/full_dataset_comparison_100/<RUN_TAG>/runs/`.
Each configuration/seed has its own directory. Old artifacts are never overwritten.
Shared raw/text files, verified model backup and frozen representation caches
may be reused. The notebook records a dependency lock for relevant packages,
full pip freeze, Python/CUDA/device, code SHA and per-run numerical identities.
Installing packages that change an already imported PyTorch requires restarting
the runtime; the notebook checks this. Colab's Python/CUDA runtime must also match.

Equivalent CLI (CUDA required):

```bash
python run_full_dataset_comparison.py --output-dir experiments/research/full_100_run1 --skip-completed --resume
```

For Drive resources provide `--raw-dir`, `--text-dir` and `--cache-dir` as in the
notebook. `--text-batch-size 256 --eval-batch-size 512` are defaults. No warmup,
epoch or learning-rate override is needed: they are fixed per configuration.

After disconnect, use the same run tag, source, packages, GPU and settings and
rerun setup and Section 6. Completed-run identities and checkpoint checksums are
verified; partial runs restore only their own atomic epoch-end `last.pt`. An
uncommitted epoch may be replayed (up to nine optimizer steps); the completed
model still has 900 committed updates. Replayed compute is not part of the
committed time/update counter. No mid-epoch resume or exact cross-device
reproducibility is claimed. Code changes require a new output directory; use the
historical commit to resume older-stage checkpoints.

Saved per run: config.json, manifest.json, best.pt, last.pt, history.json/CSV,
steps.csv, metrics.csv, results.json, and resume_audit.json when resuming. Last
checkpoint includes model/buffers, Adam state, module modes, all RNG streams,
epoch/step counters, selected model state, best MRR/epoch, history, time and GPU
peak. Last checkpoint is authoritative if a disconnect interrupts CSV writing.
The next committed epoch or completed export repairs derived tables.

Standalone baseline (no LM):

```bash
python run_full_dataset_comparison.py --single baseline --seed 0 --output-dir experiments/research/separate_full_baseline
```

Regenerate reports without training or evaluation:

```bash
python run_full_dataset_comparison.py --report-only --output-dir experiments/research/full_100_run1
```

## Output and interpretation

Every epoch prints phase/loss/elapsed/peak memory; scheduled validation also prints
MRR and Hits@10. All validation/test metrics include Hits@1/3/10, optimistic MRR,
tied-query fraction and mean/max ties. Full subprocess output and tracebacks are
streamed and saved. No child survives a completed run, releasing its GPU allocations.

The report validates all three identities, budgets, phases, initial fingerprints,
step histories and selection before comparing results. Baseline intentionally has
no LM identity; both text configurations must share one and matching cache keys.
Outputs: summary.json, runs.json, comparison.csv, runs.csv, epochs.csv, steps.csv,
ties.csv, training_behavior.json, paired_comparisons.json and
paired_seed_differences.csv and research_analysis.md. The Markdown report answers
the research questions descriptively from the validated runs and is displayed in
the notebook. Missing/mixed runs cannot produce a winner.

Thirteen PNG plots: mean_training_loss, mean_validation_mrr,
mean_validation_hits10, test_mrr, test_hits10, per_seed_loss,
per_seed_validation_mrr, training_time, gpu_memory, optimizer_steps_loss,
validation_mrr, validation_hits10, hits1_hits3. Warmup is dashed, integrated solid,
baseline dotted. No mean mixes training phases. All 900 step losses are retained.
Epoch loss preserves the mean of nine update losses; an additional example-weighted
loss accounts for the smaller final batch, without changing optimization.

Paired reports compare warmup/no-warmup and each text configuration against baseline
for validation/test MRR/Hits, duration, ranking instability and descriptive convergence.
Percent changes are null for zero denominators. Positive/negative counts describe only seed 0; they cannot establish
consistency across seeds. Ranking instability uses MRR-change sample SD over matched epochs
15..100. Convergence epoch is the first eligible validation reaching 95% of that
run's observed best, not an equal absolute target. Loss diagnostics include
phase-local slope, fluctuations and spikes. Best-before-final, best-to-final MRR
drop and falling-loss/stalled-validation flags describe possible overfitting;
they cannot establish it or statistical significance with one seed.

Total duration includes setup/cache, validation/test and checkpoint I/O; training
intervals and individual epoch times are also reported. Cache hits make later
setup cheaper. GPU peak includes preparation; it is not just steady-state graph
memory. Use these distinctions when judging the cost of text enrichment.

## Data audit, capacity and verification limits

The read-only full-data audit passed during proposal review: exact split counts,
unchanged original mappings, expected graph/inverse edges and leakage checks.
There are 41 weak components and 36 training-isolated entities; these remain
candidates. Dataset SHA256:
`e465b6f9eee2bc96d507ed6903116fe73b920e5f35163dc7d812455c10b70dfe`.
Full manifest SHA256:
`64eb297df2f1fc762044aef49efd3b6aabe781e16cab41fdce5fccfa70d34ed3`.

Full-data T4 training has **not** been executed. One expanded RGAT relation-weight
tensor is about 2.08 GiB; four layers can retain about 8.30 GiB before other
allocations/backward temporaries. T4 headroom is unverified. Pooled FP32 text is
about 43.3 MiB; one evaluation score matrix at batch 512 is 28.4 MiB. Graph
memory dominates; smaller text/eval chunks cannot solve that allocation. An
actual OOM stops the run with diagnostics. A larger GPU or separately reviewed
activation checkpointing may be needed; no automatic methodological changes.

The budget is 30x the pilot optimizer updates and ~3.76x its graph edges.
Full graph encoding repeats nine times/epoch; Python negative sampling and
evaluation filtering are additional bottlenecks. No runtime or full-data scores
are promised. Preliminary training is disabled, including CPU/synthetic/LM/resume
smokes. Only static, mocked and non-training checks run during this implementation.
Report fixtures contain fabricated numbers and are never research results.

The 100/10, seed-0 revision passed **10 non-training study checks, 0 failed**,
including the three-run budget, single-seed reports and plots, notebook schema
and mocked execution, and both historical notebook hashes. Log:
`experiments/research/full_100_seed0_checks.log` (ignored local artifact).
No training was launched for this revision.

Original implementation verification: **14 non-training checks passed, 0 failed** on the
local CPU environment, plus syntax checks for 78 Python files and `git diff
--check`. Checks cover all three child configurations, parent orchestration,
changed-setting rejection, full-data manifest constraints, both selection
schedules, cached-feature reuse without LM encoding, baseline construction
without LM, statistical/report guards, all 13 PNGs, generated research answers,
notebook cell execution with external actions mocked, default-disabled optional
training and both historical SHA256 hashes. Log:
`experiments/research/full_stage_verification.log` (ignored local artifact).
Initial checks caught a Windows path assertion and a PyTorch metadata/build-suffix
check; both were corrected. Historical notebook tests now validate the manual
artifact separately from its generator template, without regenerating it.
No optimizer steps, smoke training, new CPU/CUDA resume experiment or real-LM
forward/backward was run in this implementation phase. These checks therefore
do not establish full-data GPU feasibility or numerical training equivalence.

Non-training validation command:

```bash
python run_static_checks.py
```

Both old notebooks are SHA256-protected by tests. The pilot's saved source/output
is used as the reference. Historical generators are retired. The active
generator refuses overwriting its notebook:
`notebooks/build_full_dataset_notebook.py`; inspect `notebook()` for a reviewable
preview if future manual edits need reconciliation.

## Cleanup, preservation and future seeds

The 2026-10-11 cleanup is documented in [CLEANUP.md](CLEANUP.md). All three
notebooks remain byte-for-byte unchanged, including outputs and manual cells.
The current notebook still uses 100/10, seed 0, and the same paths and cells.
Set REPO_REF to the pushed cleanup commit and use a **fresh RUN_TAG** for new
training. Do not reuse an existing directory after a source identity change.

For older checkpoints, use their recorded matching source version. The last
pre-cleanup commit is `d9348c744da7c30c47e585de9e9ed4674a889399`;
older experiments can require earlier commits. Strict source/data/environment
checks remain active. No source hash exceptions or checkpoint migrations exist.

Default CLI and notebook runs still perform exactly three experiments with seed
0. An explicitly requested future independent seed is supported without bringing
back retired code:

```bash
python run_full_dataset_comparison.py --seed 17 --output-dir experiments/research/full_100_seed17 --skip-completed --resume
python run_full_dataset_comparison.py --seed 17 --report-only --output-dir experiments/research/full_100_seed17
```

Each invocation produces three runs and a single-seed report. Multi-seed
aggregation is not implemented; do not interpret a single-seed report as an
estimate of across-seed variance. Budgets, equations and all other defaults are
unchanged. Historical documentation and evidence remain at their existing paths;
[the archive index](archive/README.md) records recovery instructions.
