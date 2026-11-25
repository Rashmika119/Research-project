# Active protocol

Two study configurations use the same retained model implementations:

- [fb15k_baseline_300.yaml](fb15k_baseline_300.yaml): latest baseline-only study,
  full FB15K, seed0, width64, 10 negatives, 100 epochs, no warmup, validation every5;
  effective batch32768, scoring micro-batch4096, accumulation8 (protocol dim64_accum8_eval5_v4).
- [baseline_shared_c.yaml](baseline_shared_c.yaml): existing FB15k-237 comparison,
  seed0,100 epochs per model; baseline warmup0,Shared C gated warmup10.

The runner's make_config builds fixed settings; notebook checks Python/YAML
agreement before training. Editing YAML alone does not change the runner.
Protocol changes require matching code/docs and a fresh output tag.

Older settings are historical evidence in docs/archive. Use the matching notebook
linked from the root README; do not mix datasets, configurations or output folders.
