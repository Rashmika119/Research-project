"""LM -> KG projection for Phase 2.

Maps semantic representations produced by the frozen language model back
into the KG embedding space.

Current contract:
    LM hidden dimension = 768
    KG dimension = 32

This projection is trainable while the language model itself remains frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LMKGProjection(nn.Module):
    """Project LM representations back into KG embedding space.

    Input shape:
        [batch_size, lm_dim]

    Output shape:
        [batch_size, kg_dim]
    """

    def __init__(
        self,
        lm_dim: int = 768,
        kg_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        if lm_dim <= 0:
            raise ValueError("lm_dim must be > 0")

        if kg_dim <= 0:
            raise ValueError("kg_dim must be > 0")

        self.lm_dim = lm_dim
        self.kg_dim = kg_dim

        self.projection = nn.Sequential(
            nn.Linear(lm_dim, kg_dim),
            nn.LayerNorm(kg_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, lm_vectors: torch.Tensor) -> torch.Tensor:
        """Project LM vectors back into KG space."""

        if lm_vectors.dim() != 2:
            raise ValueError(
                "lm_vectors must have shape [batch_size, lm_dim]"
            )

        if lm_vectors.size(-1) != self.lm_dim:
            raise ValueError(
                f"Expected LM dimension {self.lm_dim}, "
                f"but received {lm_vectors.size(-1)}"
            )

        return self.projection(lm_vectors)