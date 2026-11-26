# Active configurations

- [Root baseline_config.yaml](../../baseline_config.yaml): full FB15K baseline,
  seed0, width64, 10 negatives, 300 epochs, no warmup, validation every5;
  effective batch32768, micro4096, accumulation8. The runner loads this YAML
  directly. Edit it and push to main; reuse notebooks/fb15k_baseline_300.ipynb.
  Use a new run.tag for changed settings. Budgets and reports are derived.
- [baseline_shared_c.yaml](baseline_shared_c.yaml): existing FB15k-237 comparison,
  seed0, 100 epochs per model; baseline warmup0, Shared C gated warmup10.
  This older comparison retains its fixed Python/YAML contract.

The old duplicated FB15K YAML was moved to the repository root. Historical
protocols in docs/archive remain evidence, not active settings.
