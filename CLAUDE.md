# Current instructions: baseline versus Shared C gated

Read README.md, docs/BASELINE_SHARED_C.md and experiments/configs/baseline_shared_c.yaml.
The latest user selected only baseline and shared-c-gated. Other implementations,
study launchers and choices are retired. docs/archive is historical evidence,
not current instructions.

Use run_final_comparison.py and notebooks/baseline_vs_shared_c_full.ipynb.
Exactly two independent scratch runs, seed0 only, complete FB15k-237,100 epochs.
Baseline:100 structural epochs, no separate warmup. Shared C:10 warmup+90 integrated,
retaining its own Adam state. Never transfer the baseline checkpoint.

Preserve width32, two RGAT layers/one head, graph dropout.2, projection dropout.1,
Adam lr.001, weight decay.00001, clipping1, four negatives, BCE positive mean plus
negative mean, float32 and strict determinism. Batch32768 gives9 updates/epoch,
last9971;900 updates/run,1800/study. Text256/length64, eval512/every5. No scheduler,
AMP, early stopping, accumulation or activation checkpointing. Preserve original
splits/mappings, train-only graph/inverses, all14541 candidates, corrected tie ranks.

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

Preserve data/caches/results/checkpoints and archived hashes. Use a fresh output
root/tag for changed source/settings. The notebook builder refuses overwrite;
preserve user outputs before updates. Do not commit/push unless requested.
Colab uses the exact pushed SHA containing this protocol.
