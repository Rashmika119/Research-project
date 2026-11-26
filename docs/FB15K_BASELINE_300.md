# FB15K baseline: one notebook, root YAML

Use `notebooks/fb15k_baseline_300.ipynb`. It fetches `main` and prints the resolved
commit. `baseline_config.yaml` in the repository root is the single settings source.
The Python runner reads it directly; it is no longer a duplicate validation file.

## Current experiment

Full FB15K (14,951 entities, 1,345 relations; 483,142/50,000/59,071 splits).
Seed 0, width 64 (32 complex dimensions), two RGAT layers, one head, dropout 0.2.
300 epochs, no warmup, 10 negatives, effective batch 32,768, micro-batch 4,096,
8 accumulation steps: 15 updates per epoch, 4,500 total. The last batch has 24,390
positives and six micro-batches. Budgets are calculated from the YAML settings.
Adam lr 0.001, weight decay 0.00001, clipping 1.0, float32, strict determinism.
Validation every 5 epochs and at the final epoch. Best validation MRR selects the
checkpoint evaluated on test. `last.pt` saves every epoch; `best.pt` on improvement.
No scheduler, early stopping, text or pretrained weights.

`runtime.disable_native_jit: true` sets `TORCH_DISABLE_NATIVE_JIT=1` before the
training subprocess imports torch, bypassing the previously failing native Triton
path. The backend setting is included in checkpoint numerical identity. The switch
is an environment-dependent workaround; local CPU checks do not certify GPU success.

## Workflow

1. Change training values under `config` in root `baseline_config.yaml`.
2. Change `run.tag` for a new experiment and update the descriptive `protocol` name.
3. Commit and push code and YAML to main.
4. Open the same notebook in Colab. Start a fresh Python session after code changes,
   then run sections 1-4 to fetch/install/verify. Section 5 launches training.
5. Results and plots use the saved run settings, not hardcoded epoch counts.

The notebook forwards `--config` to the runner. Training writes a configuration
snapshot alongside the run. Dataset location is overridden to persistent Drive.
Output tags separate experiments; no existing 100-epoch output is overwritten.
This 300-epoch study starts from scratch, not from the completed 100-epoch weights.
To resume an interrupted run, retain its tag, config, backend, packages and original
source commit (`REPO_REF` may be set to that recorded SHA). Fetching changed main
is for a new experiment, not a way to bypass strict resume identity.

```bash
python run_baseline_fb15k.py --config baseline_config.yaml --output-dir experiments/research/fb15k_300_v5 --resume --skip-completed
python run_baseline_fb15k.py --report-only --output-dir experiments/research/fb15k_300_v5
```

Report-only validates saved settings and can regenerate earlier 100-epoch reports.
The historical notebook filename is retained. Other FB15k-237 comparisons are
unchanged and must not be directly compared with these FB15K metrics.

## Completed 100-epoch evidence

The user-supplied Colab log completed with return code 0. Best epoch: 100;
validation MRR 0.18495816, Hits@10 0.34463999; test MRR 0.18612306,
Hits@10 0.34420446. Training loss 0.170458; elapsed training/validation 7,224.3s.
These are observed 100-epoch results, not predictions for the new 300-epoch run.
