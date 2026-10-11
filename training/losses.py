"""Positive and negative BCE for ComplEx; retain the established objective."""

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
