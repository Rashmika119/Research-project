"""DistMult and ComplEx scoring with triple and all-candidate interfaces.

ComplEx follows https://arxiv.org/abs/1606.06357 and uses packed real tensors.
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


class ComplExScorer(nn.Module):
    """Re(sum(h * r * conj(t))), stored as [real | imaginary].

    ``dim`` is the total real tensor width, not the complex dimension.
    A width of 32 therefore stores 16 real and 16 imaginary coordinates.
    This keeps RGAT and the existing LM bridge at their original width.
    """

    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("ComplEx requires a positive even model.dim")
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h, relation_ids, t):
        hr, hi = h.chunk(2, dim=-1)
        rr, ri = self.relation_emb(relation_ids).chunk(2, dim=-1)
        tr, ti = t.chunk(2, dim=-1)
        return ((hr * rr - hi * ri) * tr + (hr * ri + hi * rr) * ti).sum(-1)

    def score_all_tails(self, entity_emb, h, relation_ids):
        hr, hi = h.chunk(2, dim=-1)
        rr, ri = self.relation_emb(relation_ids).chunk(2, dim=-1)
        er, ei = entity_emb.chunk(2, dim=-1)
        return (hr * rr - hi * ri) @ er.t() + (hr * ri + hi * rr) @ ei.t()

    def score_all_heads(self, entity_emb, relation_ids, t):
        rr, ri = self.relation_emb(relation_ids).chunk(2, dim=-1)
        tr, ti = t.chunk(2, dim=-1)
        er, ei = entity_emb.chunk(2, dim=-1)
        return (rr * tr + ri * ti) @ er.t() + (rr * ti - ri * tr) @ ei.t()


def build_scorer(scorer_type: str, num_relations: int, dim: int):
    if scorer_type == "distmult":
        return DistMultScorer(num_relations, dim)
    if scorer_type == "complex":
        return ComplExScorer(num_relations, dim)
    raise ValueError(f"Unknown model.scorer_type: {scorer_type!r}")
