# Current instructions: approved full-data study

Read README.md, docs/FULL_DATASET_300.md and experiments/configs/full_dataset_300.yaml.
Previous instructions are archived byte-for-byte in docs/WARMUP_STAGE_INSTRUCTIONS.md.
Historical stage results/protocols remain preserved; they are not current defaults.

## Protocol

Exactly three configurations, seeds 0,1,2 (nine independent runs):
- baseline: first RGAT + ComplEx; 300 baseline epochs, 2,700 baseline updates.
- residual-no-softprompt-warmup: 30 structural + 270 integrated epochs;
  270 structural + 2,430 integrated updates.
- residual-no-softprompt-no-warmup: identical architecture; 300 integrated epochs,
  2,700 integrated updates.

Full FB15k-237: max_entities=0, 14541 entities, 237 relations,
272115/17535/20466 train/valid/test triples. Keep original mappings/splits,
training edges and inverses only, all candidates including isolated entities.
No synthetic fallback. Verify full manifest/fingerprints; reject changed data.
Negative filtering uses training facts only; evaluation filtering uses all splits.
Correct exact-tie rank: 1+higher+(equal-1)/2; optimistic MRR/ties are diagnostics.

300 epochs, training batch32768, final batch9971; nine updates/epoch. No dropping,
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
to GPU only for missing encodings. Frozen LM stays eval. Retain No-Refinement and
soft-prompt Residual outside this comparison. Do not reactivate Original variants
or delete shared mathematical/historical checkpoint inspection code.

## Evaluation

Every five epochs. New study selection_policy=scheduled_only: initial integrated
validation at epoch0/boundary30 is diagnostic only. Warmup structural validation
cannot select integrated weights. Baseline/no-warmup selection starts5; warmup
integrated selection starts35. Earliest equal best wins. Preserve the historical
initial_and_scheduled default for older workflows. After training, evaluate the
best validation checkpoint on test; select configuration by mean validation MRR,
never test. Report sample SD, paired seed effects, ties, memory/time, best/final
metrics and phase-separated learning curves. No significance claims from three seeds.
Keep epoch loss as mean update losses; additionally record example-weighted loss
and all2700 step losses, without altering optimization.

## Notebook protection and execution

NEVER modify, regenerate or revert notebooks/research_pipeline.ipynb or
notebooks/warmup_ablation_5000.ipynb. Both contain manual edits and saved outputs.
Generators must refuse overwriting them. New notebook:
notebooks/full_dataset_comparison_300.ipynb; separate generator:
notebooks/build_full_dataset_notebook.py (also refuses overwrite).
New Drive root: Research/full_dataset_comparison_300. New entrypoint:
run_full_dataset_comparison.py; --report-only never trains.
The full-study registry is separate from the retained pilot registry.

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
