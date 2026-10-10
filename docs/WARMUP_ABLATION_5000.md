# Four-configuration warmup ablation

Open `notebooks/warmup_ablation_5000.ipynb`. Its separate generator is
`notebooks/build_warmup_ablation_notebook.py`. The previous notebook is protected
and has not been regenerated, edited or replaced. The old generator can create
only a separate historical preview and refuses to overwrite an existing preview.

## Scope and exact settings

| Configuration | Architecture | Structural updates | Integrated updates |
| --- | --- | ---: | ---: |
| residual-warmup | Residual with soft prompts | 5 | 25 |
| residual-no-warmup | Identical Residual architecture | 0 | 30 |
| residual-no-softprompt-warmup | Residual with independent pooled text | 5 | 25 |
| residual-no-softprompt-no-warmup | Identical independent-text architecture | 0 | 30 |

Each configuration uses seeds 0/1/2, exactly 5,000 candidates, subset seed 0,
30 total epochs/optimizer updates, fixed Adam LR .001, weight decay .00001,
four negatives per fact, BCE, gradient clipping 1, KG width 32, two layers/one
head per RGAT, structural/refiner dropout .2, projection dropout .1, text chunk
size 4, max text length 64 and evaluation batch size 64. Validation runs every
epoch. No mixed precision, width reduction or graph-weight transfer is introduced.
RoBERTa remains frozen, pretrained and pinned to the verified existing revision.
The current YAML is `experiments/configs/research_protocol.yaml`; CLI defaults
match it, and the notebook exposes the main settings explicitly. YAML changes
alone do not automatically override CLI flags. Changing settings needs a new run tag.

`training/variants.py` separates available architectures from current experiment
configurations. No-Refinement and the independent full-data baseline still work;
neither enters this comparison. Original and Original No-Warmup are removed from
active factories/CLI choices/comparisons. Shared model code still supports historical
gates and explicit legacy checkpoint reevaluation; deleting it would break those
workflows. It is not an active experiment registration.

All trainable modules are initialized before warmup. Same-architecture warmup pairs
start with exactly matching complete trainable-state fingerprints for each seed.
All four configurations share initial structural weights for each seed. Warmup
continues with its own structural parameters/Adam moments; no-warmup starts
integrated optimization immediately. There is one full-subset update per epoch.
The two schedules intentionally distribute those 30 updates differently.

The equations are unchanged: U=E+tanh(gE)*TE; final entities=U+tanh(gF)*G2(U,A);
final relations=R+tanh(gR)*TR. Scalar raw gates start at .01. Soft-prompt TE/TR
come from KG-to-LM projection, frozen RoBERTa's injected-prompt position, and
LM-to-KG projection. Independent text uses attention-mask mean pooling of frozen
768-wide outputs followed by a trainable projection; it has no structural LM
input or KG-to-LM projection. Only frozen pooled text is cached. No-Refinement
keeps gated text fusion and omits G2/refinement gate.

## Dataset investigation and construction

The old sampler stopped retaining facts when its BFS traversal reached the entity
limit. This explains the sparse 1,022-fact old pilot. The new `train_bfs_induced_v1`
sampler uses training adjacency only, the same seeded initial root and adjacency
order, and then includes **all** original training facts between selected entities.
If a component runs out, it chooses another training-incident root deterministically
from sorted unvisited IDs. It fails if the requested count cannot be obtained from
training-incident entities. It never fills from held-out edges or silently shrinks.
Entity IDs are remapped in sorted original-ID order; relation IDs remain unchanged.
Validation/test facts are filtered independently, keeping original split boundaries.

Real FB15k-237 data were downloaded and audited locally without model training:

| Seed-0 subset | Entities | Training facts | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Historical traversal at 1,000 | 1,000 | 1,022 | 55 | 86 |
| Historical traversal at 5,000 | 5,000 | 7,221 | 4,717 | 5,465 |
| New induced graph at 5,000 | 5,000 | 72,374 | 4,717 | 5,465 |

Both 5,000-entity constructions selected exactly the same entity set. The induced
training graph has one weak component of 5,000 nodes, no isolated entities,
57,692 unique undirected pairs, average simple undirected degree 23.0768 and
density 0.0046162833 (0.4616%; self-loops and relation multiplicity excluded).
This is substantially more training information; it does not by itself prove
enough data for generalization. All twelve runs use this same construction.

See [subset_audit_5000.json](subset_audit_5000.json) for the measured data fingerprints:
dataset SHA256 `e465b6f9eee2bc96d507ed6903116fe73b920e5f35163dc7d812455c10b70dfe`;
manifest SHA256 `13fd23da9f71d36463e1af7775d847f973b5c617cdc470dfc5b119630d0a300f`.
Every child regenerates and verifies the shared manifest, including exact mappings,
facts, entity count, statistics and strategy version. Candidate scope stays 5,000.
Graph assertions verify the training facts and their inverse edges only. Negative
sampling still rejects all known original training facts within that entity scope.

To reproduce the data audit only:

```bash
python run_subset_audit.py --max-entities 5000 --subset-seed 0
```

## Launch and resume

Push this code to the repository/ref selected in the new notebook, select a T4
runtime, and run its eight sections in order. Set a stable unique RUN_TAG. The
default stage output root is `MyDrive/Research/warmup_ablation_5000`, separate from
old results. Shared raw/text data, pooled text and verified model backup can be
reused from the earlier shared resource directory. Model weights load from local
`/content/hf_cache/hub`, never by memory-mapping a Drive backup.

Section 5 separates isolated CPU regressions, pretrained validation, actual-RoBERTa
GPU smoke, repeated CUDA resume tests and actual-RoBERTa GPU resume. Both the CPU
runner and corruption-test fixtures remove inherited persistent backup settings.
Mandatory pretrained validation also protects direct execution of the launch cell.

Section 6 runs four configurations by three seeds sequentially in separate child
processes. It shows configuration/architecture/warmup/seed, epoch, phase, loss,
validation MRR/Hits@10, tracked peak GPU memory and elapsed time. Full logs remain
on Drive, including traceback/exit diagnostics. An epoch can take considerable
time before it emits its completion line.

Equivalent CLI (replace directories with Drive paths in Colab):

```bash
python run_pretrained_check.py --device cuda
python run_warmup_ablation.py --max-entities 5000 --epochs 30 --warmup-epochs 5 --seeds 0 1 2 --subset-seed 0 --output-dir experiments/research/warmup5000_run1 --skip-completed --resume
```

`run_phase5_comparisons.py` now delegates to this same four-configuration study.
To run one configuration or the separately retained architecture:

```bash
python run_phase5_training.py --configuration residual-no-warmup --max-entities 5000 --epochs 30 --output-dir experiments/research/single_no_warmup
python run_phase5_training.py --architecture no-refinement --warmup-epochs 5 --max-entities 5000 --epochs 30 --output-dir experiments/research/separate_no_refinement
python run_research_baseline.py --output-dir experiments/research/independent_baseline
```

After a disconnect, use the same code, packages, GPU policy, paths, settings and
run tag; rerun with `--skip-completed --resume`. Every run retains its own
config/manifest, `best.pt`, `last.pt`, optimizer and complete RNG states, history
JSON/CSV, metrics CSV, results JSON and optional resume audit. Model/data/seed/warmup
identity mismatches and swapped checkpoints are rejected. `last.pt` is the atomic
epoch commit; an incomplete epoch is replayed. Checkpoint fields are extended
additively with initial full-trainable-state fingerprints; RNG schema remains 2.
The new data strategy/source identity deliberately prevents resuming old-stage runs.
Existing artifacts and results are never deleted or migrated silently.

Warmup validation is structural-only diagnostic evaluation. It cannot select an
integrated checkpoint. Initial integrated evaluation occurs at the boundary
before an integrated update (epoch 5 or epoch 0) and can remain selected.

## Reports and interpretation

The complete comparison writes `comparison.csv`, `runs.csv`, `runs.json`,
`summary.json`, `training_behavior.json`, `paired_comparisons.json`,
`paired_seed_differences.csv` and `ties.csv`, alongside per-run histories/results.
Reports require all configuration/seed pairs and matching manifests, candidate
scope, numerical policy, source/environment, frozen model identity, budgets and
paired initialization. Missing/mixed experiments cannot produce a winner.

Selection uses mean validation MRR; reports then show that configuration's test
metrics. Mean/sample SD, per-seed deltas and percentage changes cover warmup effects
for each architecture and soft-prompt effects under matched warmup. Percentage
change is null when its denominator is zero. There are no p-values or claims of
statistical significance with only three seeds.

Loss analysis separates phases and reports final/minimum loss, fluctuations,
upward steps, descriptive spikes, within-phase linear slope and epoch reaching
95% of the observed within-phase reduction. A recent-window flag identifies loss
falling while MRR does not improve. These are descriptive diagnostics, not proof
of convergence or overfitting. JSON records the exact definitions/thresholds.
Per-epoch and selected-checkpoint tie diagnostics include corrected/optimistic
MRR, tied-query fraction, mean/max tied candidates and severe-tie warnings.

Eight PNG plots are generated and displayed:

1. Mean loss by epoch (`mean_training_loss.png`).
2. Mean validation MRR (`mean_validation_mrr.png`).
3. Mean validation Hits@10 (`mean_validation_hits10.png`).
4. Test MRR with sample-SD error bars (`test_mrr.png`).
5. Test Hits@10 with sample-SD error bars (`test_hits10.png`).
6. Per-seed loss (`per_seed_loss.png`).
7. Per-seed validation MRR (`per_seed_validation_mrr.png`).
8. Total execution time with sample SD (`training_time.png`).

Each configuration's means are across seeds within the same epoch/phase only.
Structural segments are dashed and integrated segments solid; shading marks the
warmup window only for enabled configurations. Warmup and integrated loss/evaluation
represent different models/objectives and must not be interpreted interchangeably.
Total time includes preparation/cache/evaluation/checkpoint overhead; compare epoch
times as well because frozen-text cache hits reduce preparation time. Timings exclude
preflight/test commands; there is no claim that they measure equal FLOPs.

## Verification and limits

Local verification for this stage: **25 CPU regression tests passed, 0 failed**,
including real-data 5,000-entity/leakage checks, all four actual 30-update schedules
on tiny fixtures, same-seed initialization, every configuration's exact resume,
swapped-checkpoint rejection, twelve CLI launches, all eight graphs, statistical
guards, phase diagnostics, notebook wiring and preservation. The first sandboxed
full-suite run passed 24 tests and hit a Windows read-permission error on the
downloaded split file; rerunning with access to those files passed all 25.

Actual pinned pretrained RoBERTa on CPU passed forward/backward, trainable-gradient
and frozen-weight checks for Residual, Residual No-Softprompt and No-Refinement.
Three additional CPU uninterrupted-control/resume repeats passed exact comparisons.
This environment uses Python 3.14.3 / PyTorch 2.11.0+cpu; no Tesla T4/live Colab was
available. **GPU execution and the twelve 5,000-entity research training runs were
not performed.** Logs are in the ignored `experiments/research/` directory:
`warmup_stage_tests.log`, `warmup_real_roberta_cpu.log`, `warmup_resume_cpu.log`;
the latter's structured report is `warmup_resume_cpu/resume_diagnostic.json`.

The original notebook SHA256 is recorded in [STAGE1_RESULTS.md](STAGE1_RESULTS.md)
and asserted by a preservation test. The new notebook has separate schema, source
syntax, generation-consistency and ordered-execution tests with mocked external
actions. Real-data subset assertions run when local FB15k-237 files exist; remaining
offline tests use synthetic fixtures with real RGAT/ComplEx and a tiny frozen LM.
No twelve-run 5,000-entity training experiment is started during development.

T4 capacity/runtime and full-stage scores require the user's Colab runs. The induced
graph is denser than the old traversal, increasing RGAT memory/compute, and soft
prompts require gradients through RoBERTa for 5,000 entity descriptions. If an actual
OOM occurs, preserve its diagnostics. Smaller text/evaluation chunks under a new
consistent run tag can reduce those respective allocations while keeping equations,
full graph and 30 updates unchanged; they cannot remove full-graph RGAT costs. A
larger GPU may be necessary. Do not silently shrink entities, widths, precision,
change the objective or substitute minibatch-update budgets.

## File inventory and cleanup

Created:

- `notebooks/warmup_ablation_5000.ipynb` and `notebooks/build_warmup_ablation_notebook.py`.
- `run_warmup_ablation.py`, `run_subset_audit.py`.
- `preprocessing/induced_subset.py`, `training/ablation_reports.py`.
- `tests/test_warmup_ablation.py`.
- `docs/WARMUP_ABLATION_5000.md`, `docs/subset_audit_5000.json`.
- `docs/STAGE1_PROTOCOL.md`, `docs/STAGE1_RESULTS.md` and
  `experiments/configs/research_protocol_1000_historical.yaml`.

Modified:

- `training/variants.py`, `training/comparisons.py`, `training/research_cli.py`,
  `training/research_pipeline.py`, `training/experiment_io.py`.
- `experiments/configs/research_protocol.yaml`.
- `run_scratch_smoke_test.py`, `run_resume_diagnostic.py`.
- `notebooks/build_colab_notebook.py` (protective output redirect only; old notebook untouched).
- `tests/test_notebook.py`, `tests/test_research_pipeline.py`, `tests/test_pretrained.py`.
- `README.md`, `CLAUDE.md`, `docs/IMPLEMENTATION.md`, `docs/RELIABILITY.md`.

Removed obsolete active Original registrations, the Original-specific warmup
exception, five-way orchestration/default configuration and tests assuming five
active architectures. Warmup pairs reuse the same model implementation. The old
traversal helper remains for provenance/audits and explicitly historical runners;
it is no longer used by current scratch data preparation. No checkpoint, result,
historical notebook or shared mathematical model file was deleted.
