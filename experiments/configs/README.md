# Root configurations

- [baseline_config.yaml](../../baseline_config.yaml): full FB15K baseline,
  AdaGrad lr0.5,batch4831,regularized logistic loss,explicit embedding L2=.01.
- [comparison_config.yaml](../../comparison_config.yaml): FB15K baseline versus
  Shared C gated; retains Adam lr.001,batch32768,micro4096x8,BCE.

Both use seed0,width64,300 epochs,10 negatives,validation every5 and native JIT
disabled. Shared C has10 warmup epochs. Edit YAML and run.tag, push main and reuse
the corresponding notebook. Training budgets are derived from settings.
