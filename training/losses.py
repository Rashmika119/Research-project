"""Loss functions.

BCE/logistic loss for DistMult/ComplEx (bilinear scorers) — CLAUDE.md
non-negotiable rule #2: margin-based ranking loss is for
translational-distance models (TransE, RotatE) only, and using it with a
bilinear scorer was the root cause of the preliminary experiment's first
diagnostic failure (README.md §15-17). Do not swap this for margin loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """pos_scores: [batch]. neg_scores: [batch, num_negatives] (or [batch])."""
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    return pos_loss + neg_loss
