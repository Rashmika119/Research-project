# Experiment register and checkpoint recovery

**Status: Phase 2 complete; Phase 3 next.** Last reconciled: 2026-10-05
(Asia/Colombo), using source tree `4b694a2`.

## Evidence conventions

- **Reported passed/completed** means retained `CLAUDE.md` notes report the
  result. Original console logs, notebooks and checkpoints were not available
  for independent verification during this reconciliation.
- **Evidence revision** identifies a commit containing the result description.
  **Code reference** identifies a relevant implementation/config snapshot.
  Neither proves which commit a historical runtime checked out.
- The **actual executed code revision and run timestamp are unknown for every
  historical run below**. Documentation commit dates are not run dates. Exact
  package versions, dataset/text checksums and dirty working-tree state were
  also not captured in the available record.
- **Expected artifact path** is the configured/reported output location, not
  a recovered local file or a confirmed Google Drive location.
- Only FB15k-237 records below belong to the active experiment series. Earlier
  WN18/WN18RR diagnostics are excluded from comparative results.

## Completed training and data gates

| ID | Entry point and config | Reported result | Expected artifact | Code reference / evidence revision |
|---|---|---|---|---|
| P0-FB | `run_phase0_smoke_test.py`; [phase0_fb15k237.yaml](configs/phase0_fb15k237.yaml) | Colab gate passed. 14,541 entities; 237 relations; train/valid/test 272,115/17,535/20,466. Toy subset: 50 entities, nonempty splits. No ranking metrics. | No checkpoint produced. Dataset: `data/raw/fb15k237/`; original gate log unavailable. | `311ef00` / `c436e81` |
| P1-RGCN-TOY | `run_phase1_smoke_test.py`; [phase1_toy.yaml](configs/phase1_toy.yaml) | Colab gate passed; early/late average loss 0.6944/0.0033. Validation MRR sequence 0.2546, 0.3174, 0.3214, 0.3169. Reload equality reported passed. | `experiments/checkpoints/phase1_toy.pt` — missing locally. | `4bac71b` / `e291909` |
| P1-RGCN-FULL | `run_phase1_full_training.py`; [phase1_full.yaml](configs/phase1_full.yaml) | Colab 100-epoch run completed. Loss 1.3855 -> 0.0948. Validation MRR 0.0511 (epoch 5) -> **0.1842 (epoch 100)**; validation Hits@10 0.1096 -> **0.3409**. Best epoch reported as 100. Hits@1/3 and test metrics not recorded. | `experiments/checkpoints/kg_only_baseline.pt` — missing locally. | `a60cf95` / `f8107d7` |
| P1-RGAT-TOY | `run_phase1_rgat_smoke_test.py`; [phase1_rgat_toy.yaml](configs/phase1_rgat_toy.yaml) | Colab gate passed after dead-parameter initialization fix. Early/late average loss 1.1134/0.1142; validation computed and reload equality passed. Exact ranking metrics not recorded. | `experiments/checkpoints/phase1_rgat_toy.pt` — missing locally. | `94faf4d` / `8299721` |

Toy metrics are software checks, not benchmark results. The R-GCN toy loss
figures are preserved verbatim from the notes; without its original log or
checkpoint, their consistency with the current loss implementation is unverified.

### Settings associated with the records

These settings come from the named configs at the code references above and
the result notes. They are not recovered runtime config dumps. Phase 1 uses
DistMult, Adam, a fixed learning rate, BCE as the sum of positive and negative
mean losses, and head/tail corruption filtered against training triples.

| Setting | P0-FB | P1-RGCN-TOY | P1-RGCN-FULL | P1-RGAT-TOY |
|---|---|---|---|---|
| Seed | 0 | 0 | 0 | 0 |
| Subset | Toy check capped at 50 entities | 50 entities, 72 training triples reported | Full dataset | 50 entities |
| Encoder / dimension | N/A | R-GCN / 32 | R-GCN / 128 | RGAT / 32 |
| Layers / dropout | N/A | 2 / 0.2 | 2 / 0.2 | 2 / 0.2 |
| Bases / heads | N/A | No basis setting / N/A | 30 / N/A | No basis setting / 2 |
| Learning rate / weight decay | N/A | 0.01 / 0 | 0.001 / 0.00001 | 0.01 / 0 |
| Epochs / batch size | N/A | 20 / 16 | 100 / 32768 | 20 / 16 |
| Negatives per positive | N/A | 4 | 4 | 4 |
| Evaluation interval / gradient clip | N/A | 5 epochs / 1.0 | 5 epochs / 1.0 | 5 epochs / 1.0 |
| Checkpoint selection | N/A | Final epoch | Best validation MRR | Final epoch |
| Synthetic fallback allowed by config | Yes; reported run used real data | Yes; reported run used real data | No | Yes; reported run used real data |

## Phase 2 bridge and alignment gates

All rows below have actual executed revision **unknown**. Their code can be
inspected at `426980e`, which contains the finalized RoBERTa/entity/relation
bridge scripts. Evidence is the detailed Phase 2 account in `4b694a2:CLAUDE.md`.
Earlier prototype bridge checks are not counted as distinct benchmark runs.
The notes do not provide a complete per-invocation log or dates.

| ID | Entry point / settings source | Reported outcome | Artifact location/status |
|---|---|---|---|
| P2-BRIDGE | `run_phase2_smoke_test.py`; [phase2_lm_toy.yaml](configs/phase2_lm_toy.yaml) | Generic bridge gate reported passed: shapes, finite output, gradient path, frozen/eval LM and fixed-input determinism. Individual execution environment/log not retained. | No checkpoint or metrics file saved by script; stdout log unavailable. |
| P2-ENTITY | `run_phase2_real_text_smoke_test.py`; constants in script | Entity bridge passed locally and on Colab T4 with real text and dummy KG vectors. Full entity text coverage reported. | No checkpoint saved; stdout logs unavailable. Text input paths below. |
| P2-RELATION | `run_phase2_relation_text_smoke_test.py`; constants in script | Relation bridge passed locally and on Colab T4 with real text and dummy relation vectors. 237/237 relations matched. | No checkpoint saved; stdout logs unavailable. Text input path below. |
| P2-SHORT-ALIGN | `run_entity_text_alignment_check.py`; constants in script | Alignment check mentioned in the completion account; separate short-only match count and individual pass log not preserved. Do not substitute the combined-text coverage for this result. | No checkpoint saved. Input `data/text/fb15k237/entity2text.txt`; absent locally. |
| P2-LONG-ALIGN | `run_long_text_alignment_check.py`; constants in script | Combined coverage reported: 14,515 long descriptions + 26 short fallbacks = 14,541/14,541; zero missing. | No checkpoint saved. Inputs `data/text/fb15k237/FB15k_mid2description.txt` and `entity2text.txt`; absent locally. |
| P2-REL-ALIGN | `run_relation_text_alignment_check.py`; constants in script | 237/237 relations matched; zero missing reported. | No checkpoint saved. Input `data/text/fb15k237/relation2text.txt`; absent locally. |

Bridge settings at the code reference: seed 0, `roberta-base`, KG dimension 32,
LM hidden size 768, projection dropout 0.1, batch size 4, maximum tokenized text
length 64 plus one KG prompt token. Both projection modules use Linear,
LayerNorm, GELU and Dropout. Real-text scripts hardcode these settings rather
than reading the generic YAML. The loss is `output.pow(2).mean()` solely to
exercise backward; no optimizer training, learned bridge checkpoint, MRR or
Hits@K is produced by these scripts. The pretrained LM revision is unpinned.

Alignment scripts use the real dataset in `data/raw/fb15k237` and preserve its
ID maps. Text resources are configured in `preprocessing/entity_text.py` and
`preprocessing/relation_text.py`; downloaded revisions/checksums were not saved.

## Failed attempts and pending runs

**Phase 3 selection (2026-10-05): R-GCN, dimension 128.**
[The integration contract](configs/phase3_integration.yaml) matches P1-RGCN-FULL:
two layers, 30 bases and graph dropout 0.2. Its bridge uses frozen
`roberta-base`, KG dimension 128 and projection dropout 0.1, with text length
64 and batch size 4 for the planned gate. This is a configuration decision,
not an additional completed run. The source checkpoint remains
`experiments/checkpoints/kg_only_baseline.pt` and is still awaiting recovery.
RGAT full training is an optional comparison track, not a prerequisite for this
selection. Historical configs and reported results above are unchanged.

| Run | Status | Artifact / revision provenance |
|---|---|---|
| R-GCN full at dimension 256, including basis-decomposition attempt | Reported CUDA OOM; not a completed baseline. | Failure history in `f8107d7:CLAUDE.md`; no recovered checkpoint or original failure log. |
| RGAT toy before dead-parameter fix | Training ran but checkpoint gate failed on non-finite `l2`; superseded by P1-RGAT-TOY. | Fix code `94faf4d`; failure account in `8299721:CLAUDE.md`. |
| RGAT full at dimension 128, heads 2 | Reported 66.44 GiB allocation OOM. | Initial config `8299721`; failure account and revised config `7ba17f6`. |
| RGAT full at dimension 32, heads 1 | Config prepared; no completed run or metrics recorded. | [phase1_rgat_full.yaml](configs/phase1_rgat_full.yaml), code reference `7ba17f6`. Intended path `experiments/checkpoints/kg_only_baseline_rgat.pt`; existence unverified. |

The revised RGAT config uses seed 0, two layers, dropout 0.2, no bases, Adam
learning rate 0.001, weight decay 0.00001, 100 epochs, batch size 32768,
four negatives, evaluation every five epochs, clip norm 1.0 and best-validation
selection. Preparing that config does not establish T4 feasibility.

## Checkpoint recovery inventory

Search performed on 2026-10-05:

- Current checkout, including ignored files: no checkpoint files or run logs.
- Enclosing research directory, including the sibling `code/research-complete`:
  no `.pt`, `.pth`, `.ckpt`, `.safetensors` or model `.bin` artifacts found.
  No relevant checkpoint archive was found by filename search.
- Available Git branches and reflog history: no tracked `.pt`, `.pth` or
  `.ckpt` artifacts. Checkpoint output directories are ignored by Git.
- Reflog README versions at `ef1386b` and `302b06f` describe reverted code;
  they are not evidence of trained checkpoints for this branch.
- Cloud/Colab storage was not searched: no durable source folder or link has
  been supplied. Access to the user-profile directory listing was denied;
  Downloads/Desktop were not inventoried. This is not a claim that all copies
  have been lost.

| Artifact | Historical creation claim | Recovered path | Source location / SHA-256 / actual revision | Recovery status |
|---|---|---|---|---|
| `phase1_toy.pt` | P1-RGCN-TOY reports save/reload | None | Unknown / unknown / unknown | Awaiting source copy |
| `kg_only_baseline.pt` | P1-RGCN-FULL reports save | None | Unknown / unknown / unknown | Awaiting source copy; primary baseline recovery priority |
| `phase1_rgat_toy.pt` | P1-RGAT-TOY reports save/reload | None | Unknown / unknown / unknown | Awaiting source copy |
| `kg_only_baseline_rgat.pt` | No successful full run recorded | None | Unknown / unknown / unknown | Unverified; do not label as a completed artifact |

### Recovery acceptance checklist

1. Locate the original Colab/Drive/download copy and its run log. Record the
   exact source location before copying into `experiments/checkpoints/`.
   Preserve originals and avoid overwriting files with the same name.
2. Record byte size and SHA-256, recovered destination, and recovery date.
   On Windows, use `Get-FileHash -Algorithm SHA256 -LiteralPath <path>`.
3. Load a trusted checkpoint on CPU with `torch.load(..., weights_only=True)`.
   The existing format contains `model_state`, `config`, `num_entities`,
   `num_relations`, and (for later saves) `best_val_mrr`.
4. Rebuild through `training.model_factory.build_model`, strictly load the
   state, and verify finite tensors, matching encoder/dimension/config and
   entity/relation counts. Full R-GCN is expected to have 14,541 entities,
   237 relations and dimension 128. Toy checkpoints need their toy ID mapping.
5. Verify dataset ordering and ID maps against the original dataset/run
   evidence before using weights. Counts alone do not establish ID alignment;
   current checkpoints do not embed maps or dataset fingerprints.
6. If data and environment are available, re-evaluate validation with the
   original protocol and compare with the recorded result. Record any
   discrepancy. Do not infer the original Git revision from a matching score.
7. Fill the inventory only with verified metadata. If no original survives,
   retrain as a **new run** with its own timestamp, exact revision, config,
   metrics, durable checkpoint and logs. A rerun is not historical recovery.

The checkpoint schema lacks optimizer state, RNG state, epoch and ID maps, so
recovering weights alone does not enable exact training resumption. Periodic
resumable checkpointing remains implementation work; this documentation update
does not change training behavior.

## Inspecting historical evidence

From the repository root, for example:

```bash
git show f8107d7:CLAUDE.md
git show a60cf95:experiments/configs/phase1_full.yaml
git show 8299721:CLAUDE.md
git show 4b694a2:CLAUDE.md
git show 426980e:run_phase2_relation_text_smoke_test.py
```

The preceding codebase review also passed syntax checks on 34 Python files,
offline preprocessing checks and five synthetic CPU optimization steps plus
evaluation for each encoder. Those were ad hoc software checks, not historical
FB15k-237 replications or additional benchmark runs; no checkpoint was saved.
