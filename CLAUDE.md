# Current instructions: shared-RGAT and preserved full-data studies

Read README.md, docs/FULL_DATASET_300.md and experiments/configs/full_dataset_100.yaml.
Previous instructions are archived byte-for-byte in docs/WARMUP_STAGE_INSTRUCTIONS.md.
Historical stage results/protocols remain preserved; they are not current defaults.

## Final approved residual versus Shared C comparison

The latest user-approved task matches the existing architectures. Use
`docs/FINAL_RESIDUAL_VS_SHARED_C.md`, `run_final_comparison.py`, and the NEW
`notebooks/final_residual_vs_shared_c_full.ipynb`. Exactly residual-no-softprompt-warmup,
shared-c-gated, shared-c-ungated; seed 0; full FB15k-237 max_entities=0;
100 epochs including 10 warmup; 900 updates/run (90 warmup + 810 integrated).
Preserve all numerical/model defaults, same-seed structural/projection pairing,
independent/shared layer identities, actual cached frozen RoBERTa and evaluation.
The final study adds diagnostics/reports, not new model mathematics. Earlier
registries remain unchanged. Protect ALL four earlier notebooks and results.
No actual training during implementation; static/mocked/derivative checks only.
New root Research/final_three_architectures_100; new seed0 run tag; no transfers.
See the final guide for exact-source resume and GPU limitations.

## Newly approved shared-RGAT study

The user's latest instruction authorizes six additional experiments. Read
`docs/SHARED_RGAT_1000.md` and `experiments/configs/shared_rgat_1000.yaml`.
Use `run_shared_rgat_comparison.py` and `notebooks/shared_rgat_1000.ipynb`.
Exactly two unique RGAT layers per model: A L1/fusion/shared L1/L2;
B L1/fusion/L2; C L1/L2/fusion/shared L1/shared L2. Each has gated and ungated
refinement. Ungated is U+V, retaining the skip and both text gates. L1 includes
ReLU/dropout at every call; L2 has no added activation. Warmup uses only the
structural path: one layer for A/B, two for C. Preserve real parameter sharing.
All six: train-induced 2000 entities, subset seed 0, model seed 0, 50 total epochs,
10 warmup + 40 integrated. Keep batch32768, lr .001, four negatives, width32,
text256/length64, eval512/every5. Report actual subset update budget.
Keep the original full-data registry and model state layout unchanged. Do not
restore retired soft-prompt architectures. New registry is SHARED_CONFIGURATIONS.
The shared-study notebook/config/guide retain their historical `_1000` filenames.
The user authorized updating the shared notebook to 2000/50/10; its new output
tag is shared_rgat_2000_50_10_seed0_v1_run1 under Research/shared_rgat_2000.
Protect other notebooks and saved results; the generator still refuses overwrite.
Validate using non-training tests only; users launch real training in Colab.

## Preserved full-dataset protocol

Exactly three configurations; approved default seed 0 (three independent runs):
- baseline: first RGAT + ComplEx; 100 baseline epochs, 900 baseline updates.
- residual-no-softprompt-warmup: 10 structural + 90 integrated epochs;
  90 structural + 810 integrated updates.
- residual-no-softprompt-no-warmup: identical architecture; 100 integrated epochs,
  900 integrated updates.

Full FB15k-237: max_entities=0, 14541 entities, 237 relations,
272115/17535/20466 train/valid/test triples. Keep original mappings/splits,
training edges and inverses only, all candidates including isolated entities.
No synthetic fallback. Verify full manifest/fingerprints; reject changed data.
Negative filtering uses training facts only; evaluation filtering uses all splits.
Correct exact-tie rank: 1+higher+(equal-1)/2; optimistic MRR/ties are diagnostics.

100 epochs, training batch32768, final batch9971; nine updates/epoch. No dropping,
accumulation, early stopping or hidden budget changes. Adam .001, weight decay
.00001, gradient clip1, four negatives, BCE, full precision, width32,
two-layer/one-head RGAT/dropout .2, projection dropout .1, text batch256,
length64, eval batch512. No scheduler/scaler. Changed chunks require explicit
configuration/new tag. Equal update counts do not mean equal computational cost.

## Mathematics and initialization

Initialize every trainable module from scratch before warmup. Never transfer
trainable weights between runs. Pair same-seed structural fingerprints across all
models and complete trainable fingerprints across text schedules. Warmup retains
only its own structural parameters and Adam moments.

Keep RGAT, ComplEx, BCE, sampling, projection and fusion/refinement equations:
U=E+tanh(gE)*TE; final entities=U+tanh(gF)*G2(U,A);
final relations=R+tanh(gR)*TR; raw scalar gates start .01.
Independent frozen RoBERTa uses masked mean pooling including unmasked special
tokens, then live Linear/LayerNorm/GELU/Dropout 768-to-32 projection. Cache only
frozen pooled vectors; never projected or structural outputs. Encode full graph
anew once per optimizer update, sharing the encoding for positive/negative scores.

Baseline must never construct RoBERTa or require text resources. Text models
construct frozen LM once per child, keep it CPU on pooled-cache hits, and move it
to GPU only for missing encodings. Frozen LM stays eval. The preserved full-data
registry contains baseline and residual-no-softprompt; the new study adds the
separate shared-layer implementation described above. Do not restore soft prompts,
No-Refinement, Original, RGCN or DistMult. Both text schedules share one class.
Retain the minimal integration base and parameter order for checkpoint keys.

## Evaluation

Every five epochs. Full-data study selection_policy=scheduled_only: initial integrated
validation at epoch0/boundary10 is diagnostic only. Warmup structural validation
cannot select integrated weights. Baseline/no-warmup selection starts5; warmup
integrated selection starts15. Earliest equal best wins. Preserve the historical
initial_and_scheduled default for older workflows. After training, evaluate the
best validation checkpoint on test; select configuration by mean validation MRR,
never test. Report seed-0 differences and unavailable across-seed SD, ties, memory/time, best/final
metrics and phase-separated learning curves. No significance claims or across-seed consistency claims from one seed.
Keep epoch loss as mean update losses; additionally record example-weighted loss
and all900 step losses, without altering optimization.

## Notebook protection and execution

NEVER modify, regenerate or revert notebooks/research_pipeline.ipynb or
notebooks/warmup_ablation_5000.ipynb. Both contain manual edits and saved outputs.
Historical generators were removed; never regenerate these notebooks. Active notebook:
notebooks/full_dataset_comparison_300.ipynb; separate generator:
notebooks/build_full_dataset_notebook.py (also refuses overwrite).
New Drive root: Research/full_dataset_comparison_100. New entrypoint:
run_full_dataset_comparison.py; --report-only never trains.
The full-study runner remains available alongside the separately authorized shared-RGAT study. Retired source
is recoverable at d9348c744da7c30c47e585de9e9ed4674a889399; historical YAMLs
are in docs/archive/configs. See docs/CLEANUP.md for the audit and verification.
Optional future seeds use --seed N with a new output directory; default stays 0.
Do not modify the active notebook or its outputs during source cleanup.

No preliminary test training by default: CPU/synthetic/real-LM/CUDA-resume flags
false. No one-epoch or smoke requirement before the real runs. Keep optional
developer tests. Lightweight file integrity, CUDA info, dataset/config/manifest
and checkpoint identity checks remain enabled. Do not launch expensive training
during implementation; use static/mocked non-training validation. Actual training
starts only when the user executes the training cell/CLI. Never claim GPU
verification based on CPU or mocked checks.

## Reliability and capacity

Pinned RoBERTa revision e2da8e2f811d1448a5b465c236feacd80ffbac7b and strict
checksums remain. HF_HOME=/content/hf_cache; consistent Hub paths; remove stale
TRANSFORMERS_CACHE before model imports. Verified Drive backups copy to local
storage. Never mmap Drive weights, clear cache trees or substitute random models.
Corruption fixtures must not inherit persistent RESEARCH_MODEL_BACKUP.

Strict deterministic algorithms remain enabled. Restore/audit own atomic last.pt:
model/buffers, Adam, module modes, Python/NumPy/Torch CPU/all-CUDA RNG, negative
sampler/shuffle, progress and selected state. Preserve RNG schema2/additive fields.
Source/data/environment/numerical mismatches reject resume. No exact cross-GPU
claim or unexplained tolerance relaxation. An unfinished epoch replays; last.pt
is authoritative over derived CSVs. Source changes need new outputs; historical
checkpoints require their historical source version.

T4 full-data memory is unverified: four edge-expanded RGAT tensors alone are
about8.3GiB before backward temporaries. Graph recomputes nine times/epoch.
OOM must stop with diagnostics; never silently change entities/width/precision,
equations/objective, training batches or update budgets. Text/eval chunks do not
remove graph allocations. Larger GPU or activation checkpointing needs explicit
consistent protocol/new tag. Preserve all historical artifacts and documents.

Non-training development verification: `python run_static_checks.py`. This runs
CPU model derivative checks but no optimizer steps or training loops. Optional
training/smoke/resume runners remain opt-in only. Never use unittest discovery
when a task forbids training: test_research_pipeline includes training tests.
