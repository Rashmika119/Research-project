"""KG Encoder Phase 1 (RGAT variant): relation-aware graph *attention* encoder.

Built as a second Phase 1 variant alongside `RGCNEncoder` (kg_encoder.py) for
a later side-by-side comparison — R-GCN stays the project's primary Phase 1
baseline (see CLAUDE.md's Phase 1 notes for why). Uses PyTorch Geometric's
`RGATConv`, which adds relation-aware attention on top of R-GCN's per-relation
weight matrices: instead of treating every neighbor under a given relation
equally, it learns how much to weight each one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv


class RGATEncoder(nn.Module):
    """Stack of RGAT layers producing structure-aware entity embeddings.

    Same shape contract as `RGCNEncoder` (kg_encoder.py): `num_message_relations`
    must be `2 * num_relations` once inverse edges are added (see
    `preprocessing.graph_builder.build_train_graph`). `heads` attention heads
    are averaged (`concat=False`), so `dim` stays the same at every layer,
    matching `RGCNEncoder`'s interface exactly — the two encoders are
    drop-in swaps for each other.
    """

    def __init__(
        self,
        dim: int,
        num_message_relations: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        heads: int = 2,
        num_bases: int | None = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList(
            [
                RGATConv(
                    dim,
                    dim,
                    num_relations=num_message_relations,
                    heads=heads,
                    concat=False,
                    num_bases=num_bases,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_type)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
