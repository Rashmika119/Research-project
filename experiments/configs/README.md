# Root configurations

Both active pipelines load YAML directly from the repository root:
- [baseline_config.yaml](../../baseline_config.yaml): baseline only, full FB15K.
- [comparison_config.yaml](../../comparison_config.yaml): baseline versus Shared C gated, full FB15K.

Both default to seed0,300 epochs,width64,10 negatives,batch32768,micro4096,
accumulation8,validation every5,native JIT disabled. Shared C uses10 warmup epochs
within the total. Edit settings and run.tag, push main, reuse the relevant notebook.
Budgets are calculated, not independently configured. Preserve old run outputs.
