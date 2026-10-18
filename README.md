# KG-only baseline versus Shared C gated

## New: FB15K baseline, 300 epochs

For the latest baseline experiment, open
[fb15k_baseline_300.ipynb](notebooks/fb15k_baseline_300.ipynb).
It fetches **main** and runs only the existing RGAT + ComplEx baseline:
full **FB15K**, seed **0**, width **100** (50 complex dimensions), **10 negatives**,
**300 epochs**, no warmup, effective batch **32,768** = **4,096 × 8** micro-batches,
**15 updates/epoch** (4,500 total),
validation every **10 epochs**, evaluation batch **512**, Adam lr **0.001**.
The model architecture and BCE/sampling equations are unchanged.
Micro-batched scoring shares one RGAT encoding per update. The final smaller
batch is weighted by actual example counts. This reduces scoring memory, not
the memory needed for full-graph RGAT activations. Use the new `accum8_v2` run tag;
old-run checkpoints require their original source/settings.

Run notebook sections 1–4 for setup and verification; **section 5 starts training**.
Commit and push these changes to main first. Results/checkpoints persist in the
new `Research/fb15k_baseline_300/` Drive folder. No RoBERTa weights are downloaded.

```bash
python run_baseline_fb15k.py --output-dir experiments/research/fb15k_baseline_300_seed0_accum8_v2 --resume --skip-completed
python run_baseline_fb15k.py --output-dir experiments/research/fb15k_baseline_300_seed0_accum8_v2 --report-only
```

See [the FB15K guide](docs/FB15K_BASELINE_300.md) and
[configuration](experiments/configs/fb15k_baseline_300.yaml). This is a separate
baseline study; the FB15k-237 two-model comparison below retains its settings.
Scores from the two datasets must not be directly compared.

## Existing FB15k-237 comparison

This repository compares exactly two knowledge-graph link-prediction models on
full FB15k-237. Both use RGAT and ComplEx. Shared C gated adds frozen RoBERTa text
features and gated refinement using the same two RGAT layers twice.

| Model | Seed | Structural warmup | Main training | Total epochs | Updates |
|---|---:|---:|---|---:|---:|
| `baseline` | 0 | None | 100 structural epochs | 100 | 900 |
| `shared-c-gated` | 0 | 10 epochs | 90 integrated epochs | 100 | 900 |

All trainable weights start from scratch independently. Shared C retains its own
warmup weights and Adam state; it never loads the baseline checkpoint. Matching
seed-0 structural initialization is verified. Equal updates do not mean equal
computation: baseline executes two graph layers per encoding; integrated Shared C
executes four, with only two unique layers.

## Run in Google Colab

Open [baseline_vs_shared_c_full.ipynb](notebooks/baseline_vs_shared_c_full.ipynb)
with a GPU runtime. Section 2 uses `REPO_REF = 'main'` to fetch the latest pushed
code from the main branch, then checks out and records its resolved commit.
Local changes must be committed and pushed to main before Colab can use them.
If main changes between sessions, existing runs still require matching source
to resume; use their recorded commit or a fresh run tag.

Run sections 1–5 for setup/verification, then section 6 for the two sequential
experiments. No preliminary training is launched. Optional CPU checks are disabled
by default. Section 7 displays comparison tables and figures; section 8 regenerates
reports from completed runs without training.

Each epoch prints phase, loss, cumulative updates, scheduled validation, timing,
and GPU memory. Outputs persist under
`MyDrive/Research/baseline_shared_c_100/baseline_shared_c_100_seed0_v1_run1/`.
Existing shared data and verified frozen-text caches can be reused. Full-dataset
Shared C capacity on a 15 GB T4 remains unverified; OOM never silently changes
the experiment settings.

## Command line

Install `requirements.txt` with a compatible CUDA PyTorch build. The research
runner requires CUDA. Use a fresh output directory for this protocol:

```bash
python run_final_comparison.py --output-dir experiments/research/baseline_shared_c_seed0 --seeds 0 --resume --skip-completed
python run_final_comparison.py --report-only --output-dir experiments/research/baseline_shared_c_seed0 --seeds 0
python run_static_checks.py
```

To run one model only, use `--single baseline --seed 0` or
`--single shared-c-gated --seed 0` with its own output directory. Baseline does
not load RoBERTa or need text resources.

Reports include validation/test MRR and Hits@1/3/10, ties, epoch and step losses,
gates, parameter counts, timing, GPU peaks, paired differences, CSV/JSON tables,
15 PNG/PDF figures, and research analysis. Epoch/model selection uses validation
only. Seed0 alone supports no significance or across-seed stability claims.

## Active code

- [Protocol and equations](docs/BASELINE_SHARED_C.md)
- [Approved settings](experiments/configs/baseline_shared_c.yaml)
- [Runner](training/final_study.py), [training/resume](training/research_pipeline.py), [reports](training/final_reports.py)
- [Baseline](models/kg_only_baseline_rgat.py), [Shared C gated](models/shared_rgat.py)
- [Verification and GPU diagnostics](docs/RELIABILITY.md)
- [Cleanup record](docs/CLEANUP.md)

Only these two models remain available. Earlier notebooks/configs/docs are
[archived evidence](docs/archive/README.md), not supported launch paths.
Datasets, cached weights, results and checkpoints were not deleted. Old
checkpoints require their original source and environment, not this new protocol.
