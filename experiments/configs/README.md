# Active protocol

Only [baseline_shared_c.yaml](baseline_shared_c.yaml) is active: full FB15k-237,
seed0,100 epochs per model. Per-model warmup: baseline0,Shared C gated10.

The runner's make_config builds fixed settings; notebook checks Python/YAML
agreement before training. Editing YAML alone does not change the runner.
Protocol changes require matching code/docs and a fresh output tag.

Older settings are historical evidence in docs/archive. Use only the notebook
linked from the root README.
