# Current instructions: baseline versus Shared C gated

## Latest approved experiment: FB15K baseline

Root YAML is authoritative; do not duplicate numeric settings in notebook/runner.
Set native-JIT policy before torch import and include it in resume identity.
Reports must use saved settings, including older 100-epoch runs.
The user approved docs/FB15K_BASELINE_300.md and
baseline_config.yaml. Use run_baseline_fb15k.py and
notebooks/fb15k_baseline_300.ipynb for this baseline-only study. Keep the existing
RGAT + ComplEx class: seed0, width64 (complex32), 2 layers, 1 head, dropout.2,
10 negatives, 300 epochs, warmup0, effective batch32768, micro4096, accumulation8
(15 updates/epoch, last24390 in six micro-batches:5*4096+3910),
4500 updates total. Validate every5 epochs with eval batch512. Adam lr.001,
weight decay.00001, clip1, float32, strict determinism; no scheduler/early stopping.
Full original FB15K counts:14951 entities,1345 relations,483142/50000/59071 splits.
Accumulate example-weighted scoring gradients on detached encoder-output leaves,
then backpropagate once through the same RGAT encoding/dropout realization. Clip
and update Adam once per effective batch. Keep full-graph activations resident;
do not claim graph-memory reduction. Protocol/run tag is dim64_nativejitoff_v5; reject old
checkpoint identities. No extra optimizer updates for micro-batches.
Explicit dataset identity must reject FB15K. No LM/text downloads for this run.
Use the new output root/tag; fetch main and record the resolved commit. Do not
alter Shared C's earlier FB15K protocol or compare cross-dataset scores.
No training during implementation; offline/mock/derivative checks only.

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
