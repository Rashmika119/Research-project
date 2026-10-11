# Active configuration

`full_dataset_100.yaml` is the sole active protocol: full FB15k-237, 100 total
epochs, 10 structural warmup epochs when enabled, seed 0, three configurations.
`training/full_study.py` constructs it; editing YAML alone does not override CLI
settings. The notebook keeps its historical `_300` filename but runs 100/10.

Historical configurations are retained byte-for-byte under
[docs/archive/configs](../../docs/archive/configs). They are not current defaults.
Optional future seeds use the main runner's `--seed N` with fresh outputs;
all architecture and optimization settings remain the same.
