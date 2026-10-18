# FB15K baseline: existing RGAT + ComplEx

Protocol: `fb15k_rgat_complex_dim100_neg10_300_seed0_accum8_v2`.
This is the existing `KGOnlyBaselineRGAT`, not a new baseline class or pure ComplEx
reproduction. Shared C remains available in the separate FB15k-237 comparison.

| Setting | Value |
|---|---|
| Dataset | Full original FB15K |
| Entities / relations | 14,951 / 1,345 |
| Train / validation / test triples | 483,142 / 50,000 / 59,071 |
| Graph | Training edges plus inverses: 966,284 edges, 2,690 message relation types |
| Seed / subset | 0 / no subset |
| Real embedding width / complex dimension | 100 / 50 |
| RGAT | Two layers, one head, no basis decomposition, dropout 0.2 |
| Activation | ReLU + dropout after L1; no added activation after L2 |
| Optimizer | Adam, lr 0.001, weight decay 0.00001, gradient clipping 1.0 |
| Negative sampling | 10 per positive; equal probability of corrupting head/tail |
| Negative filtering | Reject unchanged and known training triples; no held-out labels |
| Objective | Mean positive BCE plus mean negative BCE, unchanged |
| Epochs / warmup | 300 / 0 |
| Effective positive batch size | 32,768 |
| Scoring micro-batch size | 4,096 |
| Accumulation steps | 8 per full effective batch |
| Updates per epoch | 15: fourteen full batches and one final batch of 24,390 |
| Total updates | 4,500 |
| Validation | Full validation set every 10 epochs (30 evaluations) |
| Evaluation batch size | 512 |
| Selection | Earliest best scheduled validation filtered MRR |
| Test | Selected checkpoint, after all 300 epochs |
| Precision / determinism | Float32 / strict deterministic algorithms |
| Early stopping / scheduler / AMP | Disabled |
| Checkpoints | Atomic last.pt every epoch; best.pt on validation improvement |

## Run in Colab

Open `notebooks/fb15k_baseline_300.ipynb` and select a GPU runtime. Push the code to
main before running; the notebook fetches main and records its resolved commit.
Sections1–4 configure the environment, verify the fixed YAML/runner settings,
download/validate FB15K, and optionally run offline checks. Section5 launches one
training child. It does not run preliminary smoke training or load RoBERTa.

Persistent root:
`MyDrive/Research/fb15k_baseline_300/fb15k_baseline_300_seed0_accum8_v2_run1/`.
The child directory is `runs/baseline_seed0`; the manifest is in `runs/`.
Data lives separately from FB15k-237 under the shared cache's `data/fb15k/`.
Set a fresh RUN_TAG for a different source/configuration, not for resuming a run.

The loader uses the [RotatE repository's FB15K TSV files](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/FB15k).
All three splits are downloaded to staging and their counts validated before
publication. An existing partial directory is rejected without overwriting it.
You can instead place train.txt,valid.txt,test.txt in the configured raw directory,
using head-tab-relation-tab-tail format. There is no synthetic research fallback.
Deterministic train-first ID mapping, dataset fingerprints, all-entity candidates,
train-only graph construction and split-leakage checks are retained.

## Local GPU command

```bash
python run_baseline_fb15k.py --output-dir experiments/research/fb15k_baseline_300_seed0_accum8_v2 --resume --skip-completed
```

Use `--raw-dir /path/to/fb15k` to select existing data. The approved settings are
fixed by `training/baseline_fb15k.py` and checked against the YAML in Colab;
changing YAML alone does not alter training. The GPU requirement fails loudly
instead of silently starting CPU training.

## Progress, interruption and results

Every epoch prints loss, updates, elapsed time and allocated/reserved GPU peaks;
every tenth epoch also prints validation MRR/Hits@10. Step losses and weighted
epoch loss are saved. Notebook section6 displays committed progress even after
an interruption. It does not label partial histories as final test results.

Resume verifies source, settings, dataset, environment and numerical identity,
then restores the same model, optimizer, all RNG/sampler/shuffle state and progress.
An uncommitted epoch replays. If main changed, set REPO_REF to the recorded original
commit for resume. Earlier FB15k-237 checkpoints cannot initialize this run.

The child saves config/manifest,history.json/history.csv,steps.csv,metrics.csv,
last.pt,best.pt and results.json. The study root adds summary.json,epochs.csv,
steps.csv,metrics.csv,research_analysis.md and four PNG/PDF figures: epoch loss,
step loss,validation metrics and GPU memory. Report-only is CPU-capable and loads
saved JSON, not model checkpoints:

```bash
python run_baseline_fb15k.py --output-dir experiments/research/fb15k_baseline_300_seed0_accum8_v2 --report-only
```

## Gradient accumulation

Each effective batch samples its negatives in the original order and computes
one RGAT encoding. Scoring runs on micro-batches, each weighted by
`micro_positive_count / actual_effective_positive_count`. Gradients accumulate
on detached entity/relation output leaves; their combined gradients then flow
through the original encoder once. This retains one dropout realization for the
effective batch and avoids eight repeated full-graph encodings. Only clipping
and Adam follow this combined backward, once per effective batch.

The last update contains 24,390 positives: five micro-batches of 4,096 plus one
of 3,910. It uses six micro-batches, not eight; it is not divided by eight or
discarded. Per epoch there are 118 scoring micro-batches but still only 15
optimizer steps; across 300 epochs there are 35,400 micro-batches and 4,500
updates. Logs/step CSVs record micro-batch counts. Resume is at an epoch boundary,
so no partially accumulated gradients are checkpointed; uncommitted work replays.

This reduces scoring activation memory while retaining full-graph activations.
It is not a graph mini-batching or activation-checkpointing implementation.
Floating-point summation order changes, so gradient equivalence is numerical,
not a promise of bitwise identity with the earlier non-accumulated run. Use the
new protocol/tag instead of resuming an old checkpoint with changed settings.

## Interpretation and verification limits

This uses the requested dimension/negative/epoch increases while keeping the RGAT
architecture and existing optimization objective. It is not the original pure
ComplEx experiment and does not guarantee the paper's scores. FB15K results must
not be compared directly with FB15k-237 results. Seed0 alone supports no
across-seed stability or statistical significance claim.

Graph encoding runs once per optimizer update. GPU capacity and runtime must be
measured; 95.6GB VRAM is not proof the full RGAT computation fits. CPU negative
sampling and checkpoint I/O also contribute to runtime. The code never silently
changes batch size, precision or epoch budget after OOM. Preserve checkpoints if
the compute-unit budget is exhausted; do not claim an unmeasured completion time.

Implementation verification: offline fixture-based loader/mapping checks, wrong
dataset/budget rejection, mocked CLI/notebook execution, report generation, plus
the existing architecture/scoring/checkpoint/RNG tests. No optimizer updates or
GPU training were launched. Actual full FB15K download and GPU memory/runtime
remain unverified locally; the notebook validates real data before training.

Accumulation verification (2026-10-19): `python run_static_checks.py` ran31
checks:30 passed,1 existing FB15k-237 audit skipped because local files were
unreadable. CPU float64 derivative checks compare full-batch and micro-batched
losses/parameter gradients for full and partial windows with dropout enabled,
and verify one encoding, matching RNG consumption and no parameter updates.
This verifies the accumulation equations, not CUDA peak-memory savings or
bitwise equality between the old and new training runs.
