"""KGE scoring layer. DistMult first — simple, cheap, easy to debug (per
CLAUDE.md / README.md §3.1). ComplEx is a later addition once this baseline
is stable; don't add it speculatively before DistMult works end-to-end.

DistMult's score is symmetric in head/tail by construction
(sum(h*r*t) == sum(t*r*h)) — it genuinely cannot distinguish a relation from
its inverse. That's a known, expected limitation motivating the later switch
to ComplEx, not a bug in this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DistMultScorer(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(
        self, h: torch.Tensor, relation_ids: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """h, t: [batch, dim] entity vectors. relation_ids: [batch] long ids."""
        r = self.relation_emb(relation_ids)
        return (h * r * t).sum(dim=-1)

    def score_all_tails(
        self,
        entity_emb: torch.Tensor,
        h: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score (h, r, ?) against every entity. Returns [batch, num_entities]."""
        r = self.relation_emb(relation_ids)
        return (h * r) @ entity_emb.t()

    def score_all_heads(
        self,
        entity_emb: torch.Tensor,
        relation_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Score (?, r, t) against every entity. Returns [batch, num_entities]."""
        r = self.relation_emb(relation_ids)
        return (t * r) @ entity_emb.t()
