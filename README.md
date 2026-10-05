# Integrating Structural and Textual Semantics for Ontology-Enriched Knowledge Graph Representation Learning

BSc (Hons) Software Engineering research project, University of Kelaniya.

**Phase 3 is implemented and offline-tested. Verification with the historical
checkpoint and pretrained RoBERTa is still pending.** Updated 2026-10-05
(Asia/Colombo).

## Research objective

Test whether textual information improves KG representations for link prediction
when a frozen language model sits between two graph encoders. The primary
dataset is **FB15k-237**: 14,541 entities, 237 relations, and
272,115 / 17,535 / 20,466 train / validation / test triples in the recorded run.
The final optimization target is a KG representation.

The current implementation handles relational triples, not formal OWL or
ontology reasoning. It targets a single Colab T4-class GPU with a frozen LM.

## Architecture and status

The proposed full architecture is:

```text
KG embeddings / optional structure-only warm-up
  -> graph encoder 1
  -> KG-to-LM projection
  -> frozen RoBERTa with KG soft prompt + aligned description
  -> LM-to-KG projection
  -> graph encoder 2
  -> triple scorer and link-prediction loss
```

The selected Phase 3 encoder is **R-GCN with dimension 128**, two layers,
30 bases and dropout 0.2. It matches the reported full KG-only baseline.
The entity and relation bridge maps **128 -> 768 -> frozen roberta-base ->
768 -> 128**, using Linear, LayerNorm, GELU and dropout 0.1 in both projections.
It reads the contextual representation at the prepended soft-prompt position.

The Phase 3 gate connects restored encoder entity outputs and the scorer's
learned relation embeddings to that bridge. Graph encoder 2 and full-model
triple scoring/training remain future work. RGAT is retained as a separate
comparison; its revised full run is unverified. Historical standalone Phase 2
tests use dimension 32, while the bridge itself accepts a configurable dimension.

| Phase | Evidence and remaining work |
|---|---|
| 0: data scaffolding | Implemented; real-data gate reported passed in Colab. |
| 1: KG-only baseline | R-GCN toy/full and RGAT toy gates reported passed. Revised RGAT full run unverified. |
| 2: semantic bridge | Standalone entity/relation paths reported passed locally and on T4. |
| 3: encoder-to-bridge integration | Implemented; 12 offline tests pass. Real checkpoint / pretrained-LM gate pending. |
| 4: second graph encoder | Pending. |
| 5: full-model training | Pending end-to-end DistMult training, resumable checkpoints and test evaluation. |
| 6: comparisons and ablations | Pending text-enhanced baseline, proposed-model results, ComplEx and ablations. |

The recorded R-GCN baseline reached **validation MRR 0.1842 and Hits@10
0.3409 at epoch 100**. These are reported validation results, not test results.
Text alignment was reported as 14,515 long descriptions plus 26 short fallbacks,
and text for all 237 relations. Original checkpoint files and run logs are not
available locally; [experiments/RUNS.md](experiments/RUNS.md) records their
provenance and recovery status. No full-model accuracy improvement is established.

## Setup and offline checks

From the repository root, install the dependencies in a compatible Python,
PyTorch and PyG environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_phase3_gate.py" -v
```

The 12 tests use small trained synthetic R-GCN checkpoints, deterministic
fixture token IDs and a one-layer randomly initialized RoBERTa with hidden size
32. They exercise real gradients, nontrivial ID remapping, text alignment,
frozen weights, checkpoint rejection and failure reporting without downloads.
They do not replace the pretrained-LM gate or produce research metrics.

## Run the Phase 3 gate

Configuration: [experiments/configs/phase3_integration.yaml](experiments/configs/phase3_integration.yaml).
Entry point: [run_phase3_smoke_test.py](run_phase3_smoke_test.py).

```bash
python run_phase3_smoke_test.py
# Optional config/device override:
python run_phase3_smoke_test.py experiments/configs/phase3_integration.yaml --device cuda
```

The default checkpoint is `experiments/checkpoints/kg_only_baseline.pt`.
**It must be recovered before the real gate can run.** The runner verifies
model settings, tensor shapes, finite weights, counts and exact entity/relation
ID maps. It fails before downloads if checkpoint or identity metadata is missing.
It never substitutes random weights or overwrites the input checkpoint.

New Phase 1 saves contain `entity2id` and `relation2id`. For legacy saves, set
`phase1.id_map_path` to a verified original-run JSON sidecar:

```json
{
  "checkpoint_sha256": "SHA-256 of the checkpoint file",
  "entity2id": {"original_entity_key": 0},
  "relation2id": {"original_relation_key": 0}
}
```

Replace the example maps with the complete original maps. Counts or a newly
reconstructed ordering do not prove identity with an old checkpoint. If original
mapping evidence is unavailable, train and record a new baseline with embedded
maps; see [checkpoint recovery](experiments/RUNS.md#checkpoint-recovery-inventory).

Once prerequisites are met, the runner:

1. Restores checkpoint weights and samples up to 50 entities from training edges.
2. Copies embedding rows by entity key into toy ID order, keeping full relation
   IDs. It recomputes structural outputs on the toy graph.
3. Downloads/loads descriptions and aligns them to those same IDs.
4. Sends up to four observed entities and relations through the shared bridge
   in separate forwards/backwards. Both branches must produce finite output and
   finite, nonzero input/projection gradients. The entity branch also checks
   encoder/entity-table gradients; the relation branch checks selected scorer rows.
5. Confirms RoBERTa stays frozen and in eval mode without parameter gradients.
6. Writes a unique JSON success report under `experiments/logs/phase3/` with
   settings, provenance, hashes, remapping and gradient norms.

The squared-output loss is a gradient diagnostic. There are no optimizer steps
or benchmark metrics in this gate. Toy-graph outputs need not equal full-graph
outputs. Preserve successful reports in persistent storage when using Colab.

## Earlier phase commands

These gates remain available for regression checks. Real-data and LM scripts
may download their inputs; full training requires a GPU and can overwrite its
configured checkpoint path.

| Purpose | Command |
|---|---|
| Data gate | `python run_phase0_smoke_test.py` |
| R-GCN toy gate | `python run_phase1_smoke_test.py` |
| R-GCN full baseline | `python run_phase1_full_training.py` |
| RGAT toy / full | `python run_phase1_rgat_smoke_test.py` / `python run_phase1_rgat_full_training.py` |
| Generic standalone bridge | `python run_phase2_smoke_test.py` |
| Real entity-text bridge | `python run_phase2_real_text_smoke_test.py` |
| Real relation-text bridge | `python run_phase2_relation_text_smoke_test.py` |

Entity alignment checks (`run_entity_text_alignment_check.py` and
`run_long_text_alignment_check.py`) require text files downloaded by the real
entity-text smoke script. `run_relation_text_alignment_check.py` downloads its
relation resource when needed. Changing a Colab runtime wipes local files and
installed packages; keep checkpoint copies outside its ephemeral disk.

## Repository layout

| Location | Purpose |
|---|---|
| `preprocessing/` | Dataset, training graph, toy subset and aligned text. |
| `models/` | R-GCN/RGAT models, DistMult, frozen LM and projections. |
| `training/` | KG-only trainer, model factory, loss, negatives and Phase 3 helpers. |
| `evaluation/` | Filtered head/tail ranking. |
| `experiments/configs/` | Phase settings. |
| [experiments/RUNS.md](experiments/RUNS.md) | Historical results and artifact recovery. |
| `run_*.py` | Phase entry points and alignment checks. |
| `tests/test_phase3_gate.py` | Offline Phase 3 regression suite. |

## Remaining research work

Recover the baseline and pass the real Phase 3 gate, then add graph encoder 2
and full DistMult training. Compare KG-only, text-enhanced and proposed models
under matched settings before adding ComplEx, warm-up/dimension ablations and
multiple seeds. Measure memory/runtime before scaling the bridge to full data.

Use training-only graph edges and negative filters. DistMult uses BCE/logistic
loss; evaluation filters other known positives from all splits. Current ranking
uses optimistic ties and the trainer evaluates validation only. Final test
results, periodic checkpoints and optimizer/RNG resumption are not implemented.

Do not initialize KG tables directly from raw LM output. Earlier WN18RR
experiments with the rejected loss/setup are diagnostic history, not comparable
baselines. The nonlinear projection design also has no linear-only ablation yet.

For future runs record config, exact revision/dirty state, seed, environment,
dataset/text fingerprints, ID maps, metrics by split, selected epoch, artifact
path/hash and durable logs. Follow [CLAUDE.md](CLAUDE.md) for implementation rules.

## Documentation provenance

This overview was reconstructed from this branch's source and retained research
notes. The original long research README was not recovered. README versions in
reflog commits `ef1386b` and `302b06f` describe reverted implementations and do not
apply here. Historical evidence remains accessible through the immutable
revisions listed in the experiment register; unknown run metadata stays unknown.
