# Two-model full-dataset protocol

Protocol: `full_fb15k237_baseline_shared_c_100_seed0_v1`.
Only `baseline_seed0` and `shared-c-gated_seed0` are launched.
Settings: `experiments/configs/baseline_shared_c.yaml`.

## Architecture

L1 means RGATConv followed by ReLU and dropout(.2). L2 means RGATConv with
no added activation. Both models have two unique graph layers, one head, width32.

```text
BASELINE — KGOnlyBaselineRGAT

Random trainable entity embeddings E0
                  ↓
             L1 → L2
                  ↓
       Structural entities H
                  ↓
               ComplEx ← Trainable relation embeddings R0
                  ↓
        Triple scores → BCE loss
```

```text
SHARED C GATED — SharedStructuralRGAT + SharedRGATRefinement

Random trainable entity embeddings E0
                  ↓
             L1 → L2
                  ↓
       Structural entities H
                  ↓
    Text fusion U = H + tanh(gE) × TE
                  │
                  ├────────────────────────┐
                  ↓                        │
           SAME L1 → SAME L2                │
                  ↓                        │
           Refinement output V             │
                  ↓                        │
             tanh(gF) × V                  │
                  ↓                        │
                 ADD ←─────────────────────┘
                  ↓
      Efinal = U + tanh(gF) × V
                  ↓
               ComplEx ← Rfinal = R0 + tanh(gR) × TR
                  ↓
        Triple scores → BCE loss

Entity / relation descriptions
                  ↓
      Same frozen pretrained RoBERTa
                  ↓
      Cached masked-mean pooling (768)
                  ↓
      Same trainable projection module:
      Linear(768,32) → LayerNorm → GELU → Dropout(.1)
                  ↓
                TE / TR
```

Sharing means identical parameter objects, not copies initialized equally.
Integrated updates backpropagate through both uses of L1/L2. Each call draws
a fresh dropout mask. The skip preserves U while the refinement gate learns how
much V to add. Raw gates start at.01; initial coefficients are tanh(.01), not1.
There is no second text fusion after refinement and no soft prompt.

## Equations

Let G contain training edges plus inverses, and
`F_theta(X,G)=L2(dropout(ReLU(L1(X,G))),G)`.

```text
Baseline:
  H = F_theta(E0,G)
  Efinal = H; Rfinal = R0

Shared C gated:
  P(text) = sum_j mask_j × RoBERTa(text)_j / sum_j mask_j
  Q(p) = Dropout(GELU(LayerNorm(Wp+b)))
  TE = Q(P(entity descriptions)); TR = Q(P(relation descriptions))
  H = F_theta(E0,G)
  U = H + tanh(gE) × TE
  V = F_theta(U,G)                 # SAME theta
  Efinal = U + tanh(gF) × V
  Rfinal = R0 + tanh(gR) × TR
```

Pooling includes all unmasked tokens, including special tokens. Cache only frozen
pooled vectors; Q runs with gradients each update. Entity/relation text shares
the LM and projection. Each real32 embedding splits into real/imaginary halves
(complex width16). Both score triples using
`Re(sum_k e_h[k] × r[k] × conjugate(e_t[k]))`.
Loss is `mean(softplus(-positive_scores)) + mean(softplus(negative_scores))`.
Four negatives per positive are filtered against training facts only. Evaluation
filters facts from all splits. Average exact-tie rank is `1+higher+(equal-1)/2`.

## Training and evaluation

Full FB15k-237:14541 entities,237 relations,272115/17535/20466 train/valid/test
triples. Training graph:544230 directed edges,474 message relation types.
Preserve original mappings/splits and all candidates, including isolated entities.
No subset or synthetic research fallback.

| Setting | Baseline | Shared C gated |
|---|---|---|
| Seed | 0 | 0 |
| Epochs | 100 structural | 10 warmup + 90 integrated |
| Updates | 900 baseline | 90 warmup + 810 integrated |
| First eligible validation | 5 | 15 |
| LM/projection/gates | None | Frozen LM, live projection,3 learned gates |
| Unique RGAT layers | 2 | 2 |
| Layer calls per main encoding | 2 | 4 |

Batch32768,9 updates/epoch,last9971. Adam lr.001,weight decay.00001,clip1,
four negatives,float32,strict determinism. No scheduler/scaler,early stopping,
accumulation or activation checkpointing. Text batch256,length64; eval512/every5.
Encode the full graph anew for each update; share encoding across positive and
negative scores. Equal updates do not imply equal compute.

Shared C initializes every module before warmup. Warmup bypasses text/refinement
and trains only the structural path with ComplEx. At epoch11 the same model and
Adam continue. Never load baseline weights. Structural fingerprints must match
at scratch initialization, but weights train independently thereafter.

Warmup validation5/10 and integrated-boundary diagnostic10 cannot select Shared C
checkpoints. Earliest equal best scheduled validation MRR wins. Test only the
selected checkpoint; model selection uses validation, never test. Baseline has
more eligible validation checks; this follows the approved schedules.

## Outputs and interpretation

Each child saves config/environment identity, dataset fingerprints,last.pt,best.pt,
history.csv,metrics.csv,results.json. Parent reports include comparison/runs/epochs/
steps/evaluations tables, paired differences, summary/training-behavior JSON,
research analysis, and15 PNG/PDF plots. Gate and warmup-transition plots show only
Shared C; baseline has neither. Each epoch streams progress and diagnostics.

This compares complete models. Text,gating,refinement,warmup and compute differ;
a gain cannot be attributed solely to sharing. Seed0 supports no significance or
across-seed consistency claim. Across-seed SD is null. Synthetic test fixtures
are not research measurements. CPU checks do not replace actual GPU experiments.
