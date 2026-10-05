# Integrating Structural and Textual Semantics for Ontology-Enriched Knowledge Graph Representation Learning

BSc (Hons) Software Engineering research project, University of Kelaniya.

**Current status: Phase 2 complete; Phase 3 next.**
Record reconciled on **2026-10-05 (Asia/Colombo)** against `4b694a2`.

## Record provenance

This README is reconstructed from the current source code, configs, Git history,
and research notes in [CLAUDE.md](CLAUDE.md). The original long research README
referenced by those notes was not found. README versions in reflog commits
`ef1386b` and `302b06f`, and in the sibling `research-complete` folder, describe
later, reverted implementations; they do not describe this checkout. This file
does not claim to restore the original literature review or its numbered sections.

[The experiment register](experiments/RUNS.md) distinguishes reported historical
results from available artifacts, records config and revision provenance, and
tracks checkpoint recovery. Historical Colab runs were not rerun for this update.
No historical checkpoint has yet been recovered locally.

## Research question and scope

The central question is whether textual information can improve structural KG
representations for link prediction when a frozen language model sits between
two graph encoders. The final optimization target is a KG representation.

The primary dataset is **FB15k-237**, following the supervisor-directed move to
Freebase described in the project notes. Those notes record 14,541 entities,
237 relations, and 272,115 / 17,535 / 20,466 train / validation / test triples.
Do not compare these results directly with the earlier WN18/WN18RR experiments.

Despite the research title, the current implementation handles relational
triples; formal OWL or description-logic reasoning is outside the implemented
scope. Execution targets a single Colab T4-class GPU. Full LM fine-tuning and
ULTRA integration are outside the current plan.

## Architecture and implemented components

The proposed architecture is:

```text
Random KG embeddings / optional structure-only warm-up
  -> graph encoder 1
  -> KG-to-LM projection
  -> frozen RoBERTa with KG soft prompt + aligned description
  -> LM-to-KG projection
  -> graph encoder 2
  -> triple scorer
  -> link-prediction loss
```

Two separate paths are currently executable:

- **KG-only model:** random entity embeddings -> R-GCN or RGAT -> DistMult.
  The scorer maintains its own learned relation embeddings.
- **Standalone semantic bridge:** supplied KG vectors -> linear layer,
  LayerNorm, GELU and dropout -> frozen `roberta-base` -> projection with the
  same components back to KG space. The projected vector is prepended as a
  soft-prompt token; the contextual output at that position is used.

RoBERTa stays frozen and in evaluation mode, but its forward operations remain
in the autograd graph so gradients can reach the prompt and KG components.
Entities and relations use text aligned to the existing KG ID maps. Long entity
descriptions take priority over short text. The notes report 14,515 long
descriptions plus 26 short-text fallbacks and text for all 237 relations.

The bridge tests currently use random KG inputs, `kg_dim=32`, LM hidden size
768, batch size 4, and maximum text length 64. The bridge dimension is
configurable; 32 is a tested setting, not an architectural restriction.
R-GCN's reported full baseline uses dimension 128. RGAT's revised full config
uses dimension 32 and one head, but a successful full run is not recorded.
The encoder/dimension choice for Phase 3 remains to be resolved explicitly.

## Phase status

| Phase | Status and evidence |
|---|---|
| 0: data scaffolding | Implemented; real FB15k-237 gate reported passed in Colab. |
| 1: KG-only baseline | R-GCN toy and full runs reported passed; RGAT toy run reported passed. Revised RGAT full run remains unverified. |
| 2: semantic bridge | Complete as a standalone component; entity and relation paths reported passed locally and on T4. |
| 3: integrate encoder 1 and bridge | Next; real learned entity/relation vectors have not been connected to the bridge in this branch. |
| 4: second graph encoder | Pending assembly and standalone/integrated validation. |
| 5: full model training | Pending end-to-end DistMult training and full-data evaluation. |
| 6: comparisons and ablations | Pending text-enhanced baseline, proposed-model comparisons, ComplEx and ablations. |

The recorded full R-GCN baseline reached **validation MRR 0.1842 and Hits@10
0.3409 at epoch 100**. These are reported validation results, not test results.
There is no demonstrated link-prediction improvement from the proposed full
architecture yet. Shape/gradient smoke tests do not establish that improvement.

## Evaluation plan

Compare the KG-only baseline, a text-enhanced baseline without the full
architecture, and the proposed model with DistMult and later ComplEx. Planned
ablations include warm-up on/off and embedding dimensions; the earlier
128/200/256 plan must be reconciled with measured memory limits before runs.
Hold dataset, splits and other settings fixed for comparisons, and report
encoder and dimension differences rather than attributing them to text.

Use BCE/logistic loss for the current DistMult implementation, training-only
graph edges and negative filters, and filtered head/tail MRR and Hits@1/3/10.
Evaluation filtering uses known triples from all splits only to remove other
true candidates. The current evaluator gives optimistic ranks to ties; record
that policy when reporting results. Choose checkpoints using validation, then
evaluate the chosen model on the test split. The current trainer only runs
validation evaluation; a final test runner is still needed.

## Diagnostic history and limitations

The notes retain a preliminary WN18RR experiment using direct LM initialization
and a loss configuration later rejected by the project. Its numbers are
diagnostic history, not a valid baseline for this implementation. Preserve the
rule against replacing KG parameters with raw LM output.

The current projections include normalization, activation and dropout, beyond
the original linear-only recommendation. This is the existing implementation;
there is no recorded comparison establishing that it is better than a linear
projection. Document and evaluate this distinction when refining the method.

Full-graph encoder memory already constrained Phase 1. Four-item LM smoke tests
do not demonstrate full-dataset memory feasibility. Checkpoint recovery,
resumable training, durable logs, pinned dependency versions and dataset/text
fingerprints remain reproducibility work. The current trainer saves only at the
end of a run; a Colab disconnect can lose its in-memory best weights.

## Repository layout

| Location | Purpose |
|---|---|
| `preprocessing/` | Dataset loading, training graph, toy subset and text alignment. |
| `models/` | R-GCN/RGAT KG-only models, DistMult and standalone KG-LM bridge. |
| `training/` | KG-only training, model factory, BCE and negative sampling. |
| `evaluation/` | Filtered head/tail ranking. |
| `experiments/configs/` | Phase 0/1 configs and generic Phase 2 smoke config. |
| [experiments/RUNS.md](experiments/RUNS.md) | Historical results, settings, revision provenance and recovery inventory. |
| `run_*.py` | Phase gates, alignment checks and Phase 1 full-training entry points. |
| `notebooks/`, `tests/` | No tracked notebook or test source files in this revision; smoke scripts live at the root. |

Data, model weights and logs are runtime artifacts, not supplied with this
checkout. Expected checkpoint paths in the register are not evidence that files
are available.

## Running the existing gates

From the repository root, install `requirements.txt`. Use a compatible
PyTorch/PyG environment; record installed versions with each run. Phase 0 needs
only PyYAML; Phase 1 toy runs can use CPU. Use a GPU for full-data training and
prefer the target T4 environment for bridge verification.

```bash
python -m pip install -r requirements.txt
python run_phase0_smoke_test.py
python run_phase1_smoke_test.py
python run_phase1_rgat_smoke_test.py
python run_phase2_smoke_test.py
python run_phase2_real_text_smoke_test.py
python run_phase2_relation_text_smoke_test.py
```

The real-text smoke scripts download the required text files. After the entity
files are present, run `run_entity_text_alignment_check.py` and
`run_long_text_alignment_check.py`; `run_relation_text_alignment_check.py`
downloads relation text if needed. Downloads require network access.

Full training entry points are `run_phase1_full_training.py` and
`run_phase1_rgat_full_training.py`. They can overwrite the configured checkpoint
path: preserve recovered originals before starting a new run. Do not relabel a
new training result as a recovered historical checkpoint.

## Next milestone: Phase 3

Recover and validate the selected Phase 1 checkpoint, agree the encoder and KG
dimension, then feed actual encoder entity outputs and learned scorer relation
vectors through aligned descriptions. Keep toy/full ID mappings explicit.
The gate must show finite outputs and finite, nonzero gradients into the KG
encoder, relation embeddings and projections, with no LM parameter gradients.
Adding the second encoder and full-model training belong to later phases.

For future runs record the exact Git revision and dirty state, config, dataset
and text fingerprints, ID maps, seeds, dependency versions, device, metrics by
split, selected epoch, checkpoint path and SHA-256, durable log location, and
optimizer/RNG state for resumption. Historical unknowns remain explicitly marked
in the register rather than reconstructed as facts.
