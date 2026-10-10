# Full FB15k-237 research comparison

Current approved stage: **RGAT + ComplEx baseline**, **Residual No-Softprompt
with warmup**, and **Residual No-Softprompt without warmup**, using the complete
dataset and seeds **0,1,2**: nine independent scratch runs.

Open [the new Colab notebook](notebooks/full_dataset_comparison_300.ipynb).
Read [the complete protocol and launch/resume guide](docs/FULL_DATASET_300.md).

| Setting | Value |
| --- | --- |
| Entities / relations | 14,541 / 237 (`max_entities=0`) |
| Train / validation / test | 272,115 / 17,535 / 20,466 |
| Epochs | 300 per run |
| Warmup | 30 for the enabled text model; zero otherwise |
| Training batch / updates | 32,768; 9 per epoch; 2,700 per run |
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
Execute Section 6 when ready to launch the nine actual research runs. No smoke,
synthetic, resume-demonstration or one-epoch training runs precede them.

Default Drive stage root: `MyDrive/Research/full_dataset_comparison_300`.
Each completed epoch prints loss/phase/time/GPU peak; every fifth epoch also
prints validation MRR and Hits@10. Logs, histories and checkpoints persist.
After disconnect, keep the same code/packages/GPU/settings and RUN_TAG; completed
runs are verified/skipped and partial runs resume their own epoch-end checkpoint.
Initial/boundary validation is diagnostic only; warmup validation cannot select
integrated weights. Final test evaluation always uses the selected checkpoint.

```bash
python run_full_dataset_comparison.py --output-dir experiments/research/full_300_run1 --skip-completed --resume

# Report regeneration only; no training or model evaluation
python run_full_dataset_comparison.py --report-only --output-dir experiments/research/full_300_run1

# Non-training development checks
python -m unittest tests.test_full_study tests.test_notebook -v
```

Reports include mean/sample SD, paired seed comparisons, ties, best/final epochs,
phase-separated loss/convergence, time/GPU memory, every optimizer-step loss and
13 PNG plots. Three seeds support descriptive comparisons, not significance claims.

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

Existing pilot/full-baseline outputs and checkpoints are not deleted or replaced.
No-Refinement and soft-prompt Residual remain available through the single-model
CLI; they are excluded from this study. The pilot runner still runs the four-way
protocol, while run_research_baseline.py retains its historical 100-epoch default.
Use the new runner for the matched 300-epoch baseline. Original variants remain retired.
