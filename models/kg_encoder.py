"""KG Encoder Phase 1 / Phase 2: relation-aware graph encoder.

Starts with R-GCN (via PyTorch Geometric's `RGCNConv`) because it's the
simplest relation-aware encoder to debug, per CLAUDE.md's Phase 1 guidance.
Relational GAT / CompGCN are noted as later alternatives and are
deliberately not implemented here — don't add them speculatively before
R-GCN is proven to work end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """Stack of R-GCN layers producing structure-aware entity embeddings.

    `num_message_relations` must be the TOTAL number of distinct edge types
    used for message passing — i.e. `2 * num_relations` once inverse edges
    are added (see `preprocessing.graph_builder.build_train_graph`), NOT the
    original relation count the scorer uses. Mixing these two up is an easy
    way to get a silent shape/index-out-of-range bug.
    """

    def __init__(
        self,
        dim: int,
        num_message_relations: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_bases: int | None = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList(
            [
                RGCNConv(
                    dim,
                    dim,
                    num_relations=num_message_relations,
                    num_bases=num_bases,
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
