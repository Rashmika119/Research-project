# KG-only baseline versus Shared C gated

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
with a GPU runtime. In section 2, set `REPO_REF` to the exact **pushed commit SHA
containing this cleanup** (`git rev-parse HEAD` after committing). The notebook
checks out that commit, not a moving branch. Local edits are unavailable in Colab
until committed and pushed.

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
