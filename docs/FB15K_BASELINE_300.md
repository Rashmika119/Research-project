# FB15K baseline: one notebook, root YAML

Use `notebooks/fb15k_baseline_300.ipynb`. It fetches `main` and prints the resolved
commit. `baseline_config.yaml` in the repository root is the single settings source.
The Python runner reads it directly; it is no longer a duplicate validation file.

## Current experiment

Full FB15K, seed0,width64 (complex32),300 epochs,no warmup,10 negatives.
Effective batch4831,micro4831,accumulation1:101 updates/epoch,30300 total.
The final batch has42 positive triples. AdaGrad lr0.5, default eps1e-10,
lr_decay0, initial accumulator0; gradient clipping1.0 remains enabled.
Validation every5 epochs plus final epoch; last.pt each epoch, best.pt on improvement.
Float32, strict determinism; no scheduler/early stopping.

The explicit objective per update is:

`sum(softplus(-positive_scores)) + sum(softplus(negative_scores))`, divided by
`number_of_positive_scores + number_of_negative_scores`, then add
`0.01 * (sum(entity_embedding_table**2) + sum(relation_embedding_table**2))`.

L2 is applied once per effective update to the complete trainable input entity
and scorer relation tables, not RGAT weights or intermediate graph embeddings.
Optimizer weight decay is zero to avoid double regularization. This differs from
the former separately averaged positive/negative BCE: with10 negatives, positive
examples now carry1/11 of the data-loss weight. This is an RGAT-baseline objective,
not a claim of exact ComplEx-paper reproduction. Comparison YAML remains Adam/BCE.

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
python run_baseline_fb15k.py --config baseline_config.yaml --output-dir experiments/research/fb15k_300_adagrad_v6 --resume --skip-completed
python run_baseline_fb15k.py --report-only --output-dir experiments/research/fb15k_300_adagrad_v6
```

Report-only validates saved settings and can regenerate earlier 100-epoch reports.
The historical notebook filename is retained. Other FB15k-237 comparisons are
unchanged and must not be directly compared with these FB15K metrics.

## Completed 100-epoch evidence

The user-supplied Colab log completed with return code 0. Best epoch: 100;
validation MRR 0.18495816, Hits@10 0.34463999; test MRR 0.18612306,
Hits@10 0.34420446. Training loss 0.170458; elapsed training/validation 7,224.3s.
These are observed 100-epoch results, not predictions for the new 300-epoch run.
