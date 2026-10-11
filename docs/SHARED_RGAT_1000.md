# Six shared-RGAT experiments: 2,000 entities, 50 epochs, 10 warmup

This new study is separate from the preserved full-dataset three-model comparison.
Open `notebooks/shared_rgat_1000.ipynb`. The notebook, YAML and this guide retain
their historical `_1000` filenames; their current protocol is 2000/50/10.
The original full-data and pilot notebooks are unchanged. Previous 1000/30/5
results require their matching historical source and must not resume here.

## Protocol

- One deterministic, training-selected 2,000-entity induced FB15k-237 subset;
  subset seed 0, experiment seed 0. Keep all 237 relation IDs and all induced
  facts from their original train/validation/test split. No manufactured facts.
- Six independent scratch runs, 50 total epochs including 10 structural warmup
  and 40 integrated epochs. No baseline/checkpoint transfer between runs.
- The previous numerical defaults are retained: train batch 32768, Adam .001,
  weight decay .00001, gradient clip 1, four filtered negatives, BCE, width 32,
  RGAT one head/dropout .2, projection dropout .1, full precision/determinism,
  text batch 256, maximum length 64, evaluation batch 512.
- Updates per epoch are ceil(induced training triples / 32768). All six have
  equal optimizer budgets, with the smaller final batch retained. This is not
  an equal graph-compute budget: layer calls differ between arrangements.
- Validation every five epochs. Structural validation at epochs 5/10 and integrated
  boundary validation before epoch 11 are diagnostic only. Eligible checkpoints
  are epochs 15,20,25,30,35,40,45,50; highest corrected filtered validation MRR wins,
  earliest exact tie wins. Test is evaluated only on the selected model.
- Ranking candidates are the 2,000 selected entities. These MRR/Hits scores are
  not directly comparable with full-dataset rankings over 14,541 entities.

The local dataset audit found **6,128 train / 208 validation / 257 test triples**,
one connected component and 12,256 directed training message edges including
inverses. This means **one update per epoch, 50 per run and 300 across six runs**.
See `shared_rgat_2000_subset_audit.json` for dataset/manifest fingerprints. Colab
rebuilds and prints its own manifest and budgets from the actual input files.

## Six configurations

| ID | Structural stage (also warmup) | Refinement stage | Final entities |
| --- | --- | --- | --- |
| shared-a-gated | L1 | same L1, L2 | U + tanh(gF) V |
| shared-a-ungated | L1 | same L1, L2 | U + V |
| shared-b-gated | L1 | L2 | U + tanh(gF) V |
| shared-b-ungated | L1 | L2 | U + V |
| shared-c-gated | L1, L2 | same L1, same L2 | U + tanh(gF) V |
| shared-c-ungated | L1, L2 | same L1, same L2 | U + V |

Every model owns exactly two unique RGATConv layers, registered once under
`structural.encoder.layers`. Reuse calls the same module objects; there are no
copied encoders or duplicate optimizer parameters. Shared gradients accumulate
through all uses of each parameter.

L1 consistently means RGATConv -> ReLU -> Dropout(.2), including the standalone
first stage of A/B. L2 is RGATConv without an added activation, matching the
existing two-layer encoder's final layer. Thus C's two-layer call has the same
activation structure as the existing RGAT. Repeated L1 calls draw fresh dropout
masks in training and share weights, not masks. Dropout is disabled in evaluation.
A/B warmup leaves L2 without gradients; C warmup trains both layers. This is an
intentional distinction to consider when comparing A/B/C.

For every model, H is the structural entity output, TE/TR are projected text:

```text
U = H + tanh(gE) * TE
V = selected refinement path(U, training graph)
final entities = U + tanh(gF) * V      # gated
final entities = U + V                 # ungated
final relations = R + tanh(gR) * TR
ComplEx(final entities, final relations) -> positive/negative BCE
```

Both text gates remain trainable in all six runs. Ungated means the refinement
coefficient is fixed at 1; the residual addition is never removed. The learned
refinement gate is absent in ungated checkpoints/optimizers, and starts at raw
.01 in gated models. All common parameters start identically for the same seed;
only the presence of this scalar parameter differs between gate settings.

Independent frozen RoBERTa masked-mean vectors are cached at width 768. The
768-to-32 Linear/LayerNorm/GELU/Dropout projection remains live/trainable. No KG
soft prompts exist. Relations use gated text fusion and do not pass through the
entity graph layers. Warmup bypasses text/refinement, using structural relations.

## Colab launch

1. Push the updated source. Open `notebooks/shared_rgat_1000.ipynb` in Colab.
2. Select a GPU runtime. Set Section 2 REPO_REF to the exact pushed commit and
   choose a fresh RUN_TAG. The new notebook checks that its runner is present.
3. Run Sections 1-4: environment, Drive/dependency setup, configuration, subset
   manifest and pretrained file verification. No training occurs in those cells.
4. Optional CPU-only developer checks in Section 5 are disabled by default.
5. Run Section 6 to train all six sequentially. Every epoch streams loss, phase,
   elapsed time and peak allocated GPU memory; scheduled validation prints MRR
   and Hits@10. Subprocess errors retain full diagnostics and a persistent log.
6. Section 7 shows checkpoint status; Section 8 displays comparison tables,
   gate effects, research analysis and six PNG plots.

Default Drive root: `MyDrive/Research/shared_rgat_2000/`.
Default tag: `shared_rgat_2000_50_10_seed0_v1_run1`.
Shared raw/text resources, verified RoBERTa backups and content-keyed pooled
caches reuse the earlier resource root. Experiments use separate output folders.

```bash
python run_shared_rgat_comparison.py --output-dir experiments/research/shared_rgat_2000_50_10_run1 --skip-completed --resume
```

To run one configuration explicitly:

```bash
python run_shared_rgat_comparison.py --single shared-b-ungated --seed 0 --output-dir experiments/research/shared_b_ungated_single_run1
```

For automatic orchestration/resume, prefer the all-six command from the beginning;
it creates the parent study config/manifest before the child directories. Use a
separate folder for an independently launched single run.

Report regeneration only (no training or model evaluation):

```bash
python run_shared_rgat_comparison.py --report-only --output-dir experiments/research/shared_rgat_2000_50_10_run1
```

## Resume and results

Resume uses each run's own atomic epoch-end `last.pt`, retaining model/buffers,
Adam moments, module modes, all RNG streams, negative sampling/shuffling, progress,
selected state and histories. An uncommitted epoch replays. Source, settings,
LM/data/runtime mismatches are rejected. Do not use older study checkpoints with
new code; restore their matching historical source. No identity checks are relaxed.

Per-run artifacts: config/manifest, best.pt, last.pt, history.json/CSV, steps.csv,
metrics.csv, results.json, and a resume audit when applicable. Final reports:
summary.json, runs.json, comparison.csv, epochs.csv, steps.csv, gate_effects.csv,
research_analysis.md and six plots (loss, validation MRR/Hits@10, selected metrics,
time/memory, gate effects). Gate differences are gated minus ungated within A/B/C.
All results must pass scope/provenance, initialization, budgets, history, gate
presence and validation-selection checks before report generation.

These are single-seed descriptive results, without significance or across-seed
variance claims. Gated models start with a small refinement contribution; ungated
models apply the full contribution immediately. The comparison tests that complete
difference, including initial scaling, not only the existence of a free parameter.

## Verification

Updated-protocol verification: **31 non-training checks passed** via
`python run_static_checks.py`, including execution of all notebook cells with
external actions mocked. The real dataset subset audit also completed; its
counts above are measured from the repository's FB15k-237 files.

Run `python run_static_checks.py` for non-training CPU checks, including shared
module call counts, A/B/C warmup gradient paths, equations, cached projection
backward checks, checkpoint/RNG/Adam restoration, reports, CLI orchestration and
mocked notebook execution. Fixtures are explicitly synthetic diagnostics, never
research measurements. No optimizer step, preliminary training or actual GPU
experiment is part of this verification. GPU training and real-RoBERTa execution
for these new variants remain unverified until run in Colab.
