"""Phase 1: KG-only baseline — Encoder Phase 1 + DistMult scorer, no LM.

This is both a stepping stone toward the full KG -> LM -> KG architecture
AND one of the actual comparison rows in the final validation table
(README.md §12 — "KG-only baseline"), so it's built and checkpointed as a
real, reusable artifact, not a throwaway prototype.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.kg_encoder import RGCNEncoder
from models.scorer import DistMultScorer


class KGOnlyBaseline(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_bases: int | None = None,
    ):
        super().__init__()
        self.num_relations = num_relations

        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        # Message passing uses inverse edges too (see
        # preprocessing.graph_builder.build_train_graph), so the encoder
        # sees 2x the relation count the scorer does.
        self.encoder = RGCNEncoder(
            dim=dim,
            num_message_relations=num_relations * 2,
            num_layers=num_layers,
            dropout=dropout,
            num_bases=num_bases,
        )
        self.scorer = DistMultScorer(num_relations, dim)

    def encode(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Structure-aware embeddings for every entity: [num_entities, dim]."""
        return self.encoder(self.entity_emb.weight, edge_index, edge_type)

    def score_triples(
        self, entity_repr: torch.Tensor, triples: torch.Tensor
    ) -> torch.Tensor:
        """triples: [batch, 3] long tensor of (head, relation, tail) ids."""
        h = entity_repr[triples[:, 0]]
        r = triples[:, 1]
        t = entity_repr[triples[:, 2]]
        return self.scorer.score(h, r, t)
