# KG research experiments

The final full-dataset study compares **Residual No-Softprompt with warmup**,
**Shared C gated**, and **Shared C ungated**, using seed **0**: three scratch
runs, each **100 total epochs = 10 warmup + 90 integrated**, **900 updates**.
The requested architectures match the existing implementations.

Open [the final Colab notebook](notebooks/final_residual_vs_shared_c_full.ipynb)
and read [the final protocol and run/resume guide](docs/FINAL_RESIDUAL_VS_SHARED_C.md).
Push the code and set the notebook's exact `REPO_REF` commit before running.
Only Section 6 starts training. GPU capacity remains unverified.

```bash
python run_final_comparison.py --output-dir experiments/research/final_three_run1 --seeds 0 --skip-completed --resume
python run_final_comparison.py --report-only --output-dir experiments/research/final_three_run1 --seeds 0
```

Earlier workflows below remain preserved, with their own settings and outputs.

The new study runs **six shared-RGAT experiments** on a **2,000-entity subset**,
with **50 total epochs, 10 warmup epochs, and seed 0**. A reuses layer 1, B splits
the two layers around text fusion, and C reuses both layers. Each is tested with
and without the refinement gate; the residual addition remains in both cases.

Open [the six-experiment Colab notebook](notebooks/shared_rgat_1000.ipynb) and
read [the shared-RGAT guide](docs/SHARED_RGAT_1000.md). Push this source and set
REPO_REF to its exact commit before running Colab. Use a fresh RUN_TAG.
The notebook retains its historical `_1000` filename; it now runs 2,000 entities
and 10 warmup + 40 integrated epochs in all six experiments.

```bash
python run_shared_rgat_comparison.py --output-dir experiments/research/shared_rgat_2000_50_10_run1 --skip-completed --resume
```

The earlier full-dataset workflow below is preserved with its existing settings.

Preserved full-dataset stage: **RGAT + ComplEx baseline**, **Residual No-Softprompt
with warmup**, and **Residual No-Softprompt without warmup**, using the complete
dataset and seed **0** only: three independent scratch runs.

Open [the preserved full-dataset Colab notebook](notebooks/full_dataset_comparison_300.ipynb).
Read [the complete protocol and launch/resume guide](docs/FULL_DATASET_300.md).

| Setting | Value |
| --- | --- |
| Entities / relations | 14,541 / 237 (`max_entities=0`) |
| Train / validation / test | 272,115 / 17,535 / 20,466 |
| Epochs | 100 per run |
| Warmup | 10 for the enabled text model; zero otherwise |
| Training batch / updates | 32,768; 9 per epoch; 900 per run |
| Optimizer / LR / negatives | Adam / .001 / four per positive |
| KG width / text length | 32 / 64 |
| Text / evaluation batch | 256 / 512 |
| Validation | Every five epochs; scheduled main-model checkpoints only |
| Selection | Highest mean corrected filtered validation MRR |

All trainable components start from scratch. Only verified frozen RoBERTa weights
and independent frozen pooled text features may be reused. Baseline uses no LM.
Full training graph encoding is recomputed each update. Model equations, loss,
negative sampling and ranking are unchanged.

## Run in Colab

Push this code and set the new notebook's REPO_REF to the exact commit. Select a
T4 runtime and choose a new RUN_TAG. Run Sections 1-4 for setup and non-training
integrity checks. Optional training tests in Section 5 are **disabled by default**.
Execute Section 6 when ready to launch the three actual research runs. No smoke,
synthetic, resume-demonstration or one-epoch training runs precede them.

Default Drive stage root: `MyDrive/Research/full_dataset_comparison_100`.
Each completed epoch prints loss/phase/time/GPU peak; every fifth epoch also
prints validation MRR and Hits@10. Logs, histories and checkpoints persist.
After disconnect, keep the same code/packages/GPU/settings and RUN_TAG; completed
runs are verified/skipped and partial runs resume their own epoch-end checkpoint.
Initial/boundary validation is diagnostic only; warmup validation cannot select
integrated weights. Final test evaluation always uses the selected checkpoint.

```bash
python run_full_dataset_comparison.py --output-dir experiments/research/full_100_run1 --skip-completed --resume

# Report regeneration only; no training or model evaluation
python run_full_dataset_comparison.py --report-only --output-dir experiments/research/full_100_run1

# Non-training development checks
python run_static_checks.py
```

Reports include seed-0 metrics and differences (across-seed SD unavailable), ties, best/final epochs,
phase-separated loss/convergence, time/GPU memory, every optimizer-step loss and
13 PNG plots. One seed supports descriptive comparisons only; no across-seed variability or significance claim.

**Full-data GPU runs have not been executed during implementation.** T4 graph
memory capacity remains unverified; no automatic batch/entity/width/precision
reduction will hide an OOM. See the guide for the allocation analysis.

## Preserved research history

- [Manually edited pilot notebook](notebooks/warmup_ablation_5000.ipynb), unchanged.
- [Original notebook](notebooks/research_pipeline.ipynb), unchanged.
- [Saved single-seed pilot results](docs/STAGE2_RESULTS.md).
- [5,000-entity protocol](docs/WARMUP_ABLATION_5000.md).
- [Archived stage README](docs/WARMUP_STAGE_README.md) and [instructions](docs/WARMUP_STAGE_INSTRUCTIONS.md).
- [1,000-entity results](docs/STAGE1_RESULTS.md) and [protocol](docs/STAGE1_PROTOCOL.md).
- [Earlier implementation](docs/IMPLEMENTATION.md), [reliability investigation](docs/RELIABILITY.md)
  and [historical notes](docs/HISTORICAL_NOTES.md).

Existing outputs and checkpoints are preserved. Retired model implementations,
training entrypoints and pilot report pathways have been removed. Historical
notebooks are evidence; execute them only with their matching historical source.
Historical YAMLs are preserved under [docs/archive/configs](docs/archive/configs).
See the [cleanup and compatibility report](docs/CLEANUP.md) and
[history/recovery index](docs/archive/README.md).

The preserved full-dataset entrypoint is `run_full_dataset_comparison.py`. It exposes
`baseline`, `residual-no-softprompt-warmup`, and
`residual-no-softprompt-no-warmup`. The default remains seed 0, three runs.
A future independent seed uses `--seed N` and a new output directory; this still
runs the same three configurations with the same 100/10 budgets. Pass the same
seed when resuming or regenerating that study's reports.

Cleanup changes the source identity. Use a fresh RUN_TAG for new runs; resume
existing runs with the exact source revision saved in their provenance. The
pre-cleanup revision is `d9348c744da7c30c47e585de9e9ed4674a889399`.
Checkpoint integrity checks have not been relaxed.
