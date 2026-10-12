# Active configuration

`full_dataset_100.yaml` is the preserved full-dataset protocol: full FB15k-237, 100 total
epochs, 10 structural warmup epochs when enabled, seed 0, three configurations.
`training/full_study.py` constructs it; editing YAML alone does not override CLI
settings. The notebook keeps its historical `_300` filename but runs 100/10.

Historical configurations are retained byte-for-byte under
[docs/archive/configs](../../docs/archive/configs). They are not current defaults.
Optional future seeds use the main runner's `--seed N` with fresh outputs;
all architecture and optimization settings remain the same.

`shared_rgat_1000.yaml` documents the separately approved six-run A/B/C gate
study: 2000 entities, 50 epochs including 10 warmup, seed 0. The YAML keeps
its historical `_1000` filename. Its runner is
`run_shared_rgat_comparison.py`; see [the guide](../../docs/SHARED_RGAT_1000.md).

`final_residual_vs_shared_c.yaml` is the final full-data three-run comparison:
Residual No-Softprompt warmup / Shared C gated / Shared C ungated, seed 0,
100 total epochs including 10 warmup, 900 optimizer updates each. Runner:
`run_final_comparison.py`. See [the final guide](../../docs/FINAL_RESIDUAL_VS_SHARED_C.md).
Earlier config files remain unchanged.
