# Current research instructions

## Baseline-only FB15K experiment

Root baseline_config.yaml is authoritative; use run_baseline_fb15k.py and
notebooks/fb15k_baseline_300.ipynb. Full FB15K,seed0,width64,300 epochs,no warmup,
10 negatives,batch4831,micro4831,accumulation1:101 steps/epoch,30300 total,last42.
AdaGrad lr0.5,weight_decay0,clip1,float32,deterministic; no scheduler/early stopping.
Loss: mean softplus(-y*score) over all positive/negative examples, plus .01 times
squared norms of complete trainable entity and relation tables, once/update.
Do not regularize RGAT weights with this explicit L2. Validation every5,eval512.
Preserve the native-JIT workaround before torch import and resume identity.
Changed settings require a new run tag. Keep prior outputs. No research training
in implementation checks; use offline derivatives/mocks only.
This is not exact plain ComplEx paper reproduction. Comparison settings stay separate.

## Retained FB15K comparison

Read README.md, docs/BASELINE_SHARED_C.md and comparison_config.yaml.
The latest user selected only baseline and shared-c-gated. Other implementations,
study launchers and choices are retired. Historical output JSON in experiments/historical_outputs is evidence, not code.

Use run_final_comparison.py and notebooks/baseline_vs_shared_c_full.ipynb.
Exactly two independent scratch runs, seed0 only, complete FB15K,300 epochs.
Baseline:300 structural epochs, no separate warmup. Shared C:10 warmup+290 integrated,
retaining its own Adam state. Never transfer the baseline checkpoint.

Preserve width64, two RGAT layers/one head, graph dropout.2, projection dropout.1,
Adam lr.001, weight decay.00001, clipping1, ten negatives, BCE positive mean plus
negative mean, float32 and strict determinism. Batch32768 gives15 updates/epoch,
last24390;4500 updates/run,9000/study. Text256/length64, eval512/every5. No scheduler,
AMP, early stopping, activation checkpointing; scoring uses micro4096 and accumulation8. Preserve original
splits/mappings, train-only graph/inverses, all14951 candidates, corrected tie ranks.

Shared C: H=L2(dropout(ReLU(L1(E0)))); U=H+tanh(gE)*TE;
V=SAME_L2(dropout(ReLU(SAME_L1(U)))); Efinal=U+tanh(gF)*V;
Rfinal=R0+tanh(gR)*TR. Two unique layers, four integrated layer calls. Raw gates.01.
Frozen independent RoBERTa masked-mean pooling feeds one live entity/relation
projection. Cache frozen pooling only. Baseline must never require LM/text.
Keep retained parameter names/initialization order. No soft prompts or other models.

Scheduled validation selects checkpoints: baseline from5, Shared C from15.
Warmup5/10 and boundary-integrated10 are diagnostic only. Earliest equal best
wins; test never selects. Baseline has no projection identity, gates or integrated
boundary metric. Report budgets, losses, cost and seed0 differences without
significance or seed-consistency claims.

Maintain atomic epoch checkpoints and strict source/data/settings/environment
identity, full model/optimizer/RNG/sampler/shuffle/module-mode restoration.
No tolerance relaxation to hide resume errors. Verified pretrained weights must
load completely; no random or synthetic fallback for research.

During edits run only non-training checks: static/mocked orchestration, CPU
derivatives and fabricated checkpoint restoration. Do not launch optimizer steps
or actual training unless explicitly requested. GPU diagnostics are opt-in.
CPU checks do not certify CUDA determinism, real RoBERTa or T4 capacity.

Preserve data/caches/results/checkpoints and historical output JSON. Use a fresh output
root/tag for changed source/settings. The notebook builder refuses overwrite;
preserve user outputs before updates. Do not commit/push unless requested.
Colab defaults to fetching main and records the resolved commit. Resume still
requires matching source; use the recorded commit if main has changed.
