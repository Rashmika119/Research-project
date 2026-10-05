"""Offline integration tests using trained fixture weights and tiny random RoBERTa.

These exercise real PyTorch/PyG/transformer gradients, not pretrained-LM quality.
Run: python -m unittest discover -s tests -p "test_phase3_gate.py" -v
"""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from transformers import RobertaConfig, RobertaModel

from models.kg_lm_bridge import KGLMBridge
from preprocessing.dataset import KGDataset
from preprocessing.entity_text import align_entity_texts
from preprocessing.graph_builder import build_train_graph
from training.losses import bce_loss
from training.model_factory import build_model
from training.negative_sampling import sample_negatives
from training.phase3_gate import (
    check_bridge_paths, gradient_norm, prepare_toy_model, read_checkpoint, sha256_file,
)
from run_phase3_smoke_test import run
from training.train_kg_baseline import run as train_baseline


class FixtureTokenizer:
    """Deterministic token IDs for fixture descriptions; no downloaded tokenizer."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        ids = [[0, 4 + sum(text.encode()) % 50, 2] for text in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), 3, dtype=torch.long)}


def tiny_roberta(*args, **kwargs):
    return RobertaModel(RobertaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=48, max_position_embeddings=80,
    ))


class Phase3GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.config = yaml.safe_load(Path("experiments/configs/phase3_integration.yaml").read_text())
        cls.dataset = KGDataset(
            entity2id={f"e{i}": i for i in range(12)},
            relation2id={f"r{i}": i for i in range(3)},
            train=[(i, i % 3, (i + 1) % 12) for i in range(12)],
            valid=[(0, 1, 5)], test=[(2, 2, 8)],
        )
        model = build_model(cls.config["model"], 12, 3)
        graph = build_train_graph(cls.dataset.train, 12, 3)
        edge_index = torch.tensor(graph.edge_index).t()
        edge_type = torch.tensor(graph.edge_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        positives = torch.tensor(cls.dataset.train)
        for _ in range(3):
            optimizer.zero_grad()
            entities = model.encode(edge_index, edge_type)
            negatives = sample_negatives(positives, 12, set(cls.dataset.train), 2, random.Random(0))
            loss = bce_loss(model.score_triples(entities, positives),
                            model.score_triples(entities, negatives.reshape(-1, 3)))
            loss.backward()
            optimizer.step()
        cls.checkpoint = {
            "model_state": copy.deepcopy(model.state_dict()),
            "config": {"model": cls.config["model"]},
            "num_entities": 12, "num_relations": 3,
            "entity2id": cls.dataset.entity2id, "relation2id": cls.dataset.relation2id,
        }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "fixture.pt"
        torch.save(self.checkpoint, self.path)
        self.cp, self.maps, _ = read_checkpoint(self.path)

    def prepare(self):
        return prepare_toy_model(self.cp, self.maps, self.dataset,
                                 self.config["model"], max_entities=5, seed=3)

    def texts(self, toy):
        # File order is deliberately different from both KG and toy IDs.
        long = self.root / "long.txt"
        short = self.root / "short.txt"
        long.write_text("".join(f'e{i}\t"Description e{i}"@en\n' for i in reversed(range(12))), encoding="utf-8")
        short.write_text("", encoding="utf-8")
        return align_entity_texts(toy.entity2id, long, short).texts_by_id

    def bridge(self):
        with patch("models.frozen_lm.AutoModel.from_pretrained", side_effect=tiny_roberta):
            return KGLMBridge(**self.config["bridge"])

    def test_identity_gradients_and_frozen_weights(self):
        model, toy, full_ids = self.prepare()
        self.assertTrue(any(local != full for local, full in enumerate(full_ids)))
        self.assertTrue(torch.equal(model.entity_emb.weight,
                                   self.cp["model_state"]["entity_emb.weight"][full_ids]))
        self.assertTrue(torch.equal(model.scorer.relation_emb.weight,
                                   self.cp["model_state"]["scorer.relation_emb.weight"]))
        self.assertEqual(toy.relation2id, self.dataset.relation2id)
        names = sorted(toy.entity2id, key=toy.entity2id.get)
        for h, r, t in toy.train:
            self.assertIn((full_ids[h], r, full_ids[t]), self.dataset.train)
        texts = self.texts(toy)
        self.assertEqual(texts, [f"Description {name}" for name in names])
        bridge, tokenizer = self.bridge(), FixtureTokenizer()
        original = {k: v.clone() for k, v in bridge.lm.state_dict().items()}
        result = check_bridge_paths(model, bridge, tokenizer, toy, texts,
                                    [f"Relation r{i}" for i in range(3)])
        self.assertEqual(tokenizer.calls[0], [texts[i] for i in result["entity"]["ids"]])
        self.assertEqual(tokenizer.calls[1], [f"Relation r{i}" for i in result["relation"]["ids"]])
        for branch in result.values():
            self.assertEqual(branch["output_shape"][1], 128)
            self.assertTrue(all(norm > 0 for norm in branch["gradient_norms"].values()))
        self.assertFalse(bridge.lm.lm.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in bridge.lm.parameters()))
        self.assertTrue(all(torch.equal(v, bridge.lm.state_dict()[k]) for k, v in original.items()))

    def test_missing_checkpoint_fails_before_downloads(self):
        config = copy.deepcopy(self.config)
        config["phase1"]["checkpoint_path"] = str(self.root / "missing.pt")
        with patch("run_phase3_smoke_test.load_dataset") as loader:
            with self.assertRaisesRegex(FileNotFoundError, "trained Phase 1 checkpoint"):
                run(config)
            loader.assert_not_called()

    def test_equal_counts_reordered_ids_rejected(self):
        for key in ("entity2id", "relation2id"):
            with self.subTest(key=key):
                maps = copy.deepcopy(self.maps)
                names = list(maps[key])[:2]
                maps[key][names[0]], maps[key][names[1]] = maps[key][names[1]], maps[key][names[0]]
                with self.assertRaisesRegex(ValueError, "identity/order"):
                    prepare_toy_model(self.cp, maps, self.dataset, self.config["model"], 5, 3)

    def test_wrong_dimension_or_encoder_rejected(self):
        for key, value in (("dim", 32), ("encoder_type", "rgat")):
            cp = copy.deepcopy(self.cp)
            cp["config"]["model"][key] = value
            with self.assertRaisesRegex(ValueError, "encoder settings"):
                prepare_toy_model(cp, self.maps, self.dataset, self.config["model"], 5, 3)

    def test_nonfinite_checkpoint_rejected(self):
        self.cp["model_state"]["entity_emb.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.prepare()

    def test_legacy_checkpoint_requires_bound_sidecar(self):
        legacy = {k: v for k, v in self.cp.items() if k not in ("entity2id", "relation2id")}
        torch.save(legacy, self.path)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            read_checkpoint(self.path)
        sidecar = self.root / "ids.json"
        metadata = {**self.maps, "checkpoint_sha256": "wrong"}
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "sha256"):
            read_checkpoint(self.path, sidecar)
        metadata["checkpoint_sha256"] = sha256_file(self.path)
        sidecar.write_text(json.dumps(metadata))
        _, restored, _ = read_checkpoint(self.path, sidecar)
        self.assertEqual(restored, self.maps)

    def test_new_baseline_save_preserves_id_maps(self):
        config = yaml.safe_load(Path("experiments/configs/phase1_full.yaml").read_text(encoding="utf-8"))
        config["training"]["epochs"] = 1
        config["checkpoint_path"] = str(self.root / "new_baseline.pt")
        with patch("training.train_kg_baseline.load_dataset", return_value=self.dataset):
            train_baseline(config)
        cp, maps, _ = read_checkpoint(config["checkpoint_path"])
        self.assertEqual(maps, self.maps)
        model, toy, ids = prepare_toy_model(cp, maps, self.dataset, self.config["model"], 5, 3)
        self.assertTrue(torch.equal(model.entity_emb.weight, cp["model_state"]["entity_emb.weight"][ids]))

    def test_each_detached_branch_is_detected(self):
        for kind in ("entity", "relation"):
            with self.subTest(kind=kind):
                model, toy, _ = self.prepare()
                bridge = self.bridge()
                original = bridge.forward
                calls = []

                def broken(vectors, *args):
                    current = "entity" if not calls else "relation"
                    calls.append(current)
                    return original(vectors.detach() if current == kind else vectors, *args)

                with patch.object(bridge, "forward", side_effect=broken):
                    with self.assertRaisesRegex(AssertionError, f"{kind}: .*input gradient"):
                        check_bridge_paths(model, bridge, FixtureTokenizer(), toy,
                                           self.texts(toy), ["a", "b", "c"])

    def test_missing_text_is_rejected(self):
        model, toy, _ = self.prepare()
        with self.assertRaisesRegex(ValueError, "Missing selected entity"):
            check_bridge_paths(model, self.bridge(), FixtureTokenizer(), toy,
                               [""] * toy.num_entities, ["a", "b", "c"])

    def test_zero_and_nonfinite_gradients_are_rejected(self):
        module = torch.nn.Linear(2, 2)
        for p in module.parameters():
            p.grad = torch.zeros_like(p)
        with self.assertRaisesRegex(AssertionError, "zero gradient"):
            gradient_norm(module, "fixture")
        module.weight.grad[0, 0] = float("inf")
        with self.assertRaisesRegex(AssertionError, "non-finite gradient"):
            gradient_norm(module, "fixture")

    def test_trainable_lm_is_rejected(self):
        model, toy, _ = self.prepare()
        bridge = self.bridge()
        next(bridge.lm.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(AssertionError, "RoBERTa must be frozen"):
            check_bridge_paths(model, bridge, FixtureTokenizer(), toy,
                               self.texts(toy), ["a", "b", "c"])

    def test_runner_report_and_no_checkpoint_overwrite(self):
        config = copy.deepcopy(self.config)
        config["phase1"]["checkpoint_path"] = str(self.path)
        config["text"]["text_dir"] = str(self.root)
        config["report_dir"] = str(self.root / "reports")
        config["toy_subset"]["max_entities"] = 5
        (self.root / "entity2text.txt").write_text("".join(f"e{i}\tEntity {i}\n" for i in range(12)))
        (self.root / "FB15k_mid2description.txt").write_text("".join(f"e{i}\tLong entity {i}\n" for i in range(12)))
        (self.root / "relation2text.txt").write_text("".join(f"r{i}\tRelation {i}\n" for i in range(3)))
        digest = sha256_file(self.path)
        with patch("run_phase3_smoke_test.load_dataset", return_value=self.dataset), \
             patch("models.frozen_lm.AutoModel.from_pretrained", side_effect=tiny_roberta), \
             patch("run_phase3_smoke_test.AutoTokenizer.from_pretrained", return_value=FixtureTokenizer()):
            report = run(config, device="cpu")
        self.assertEqual(sha256_file(self.path), digest)
        self.assertEqual(report["checkpoint"]["sha256"], digest)
        files = list((self.root / "reports").glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(json.loads(files[0].read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
