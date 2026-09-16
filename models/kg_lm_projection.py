"""KG -> LM projection for Phase 2.

Maps structure-aware KG vectors from KG embedding space into the
hidden space expected by the frozen language model.

Current contract:
    KG dimension = 32
    LM hidden dimension = 768

The projection itself is trainable even though the language model is frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KGLMProjection(nn.Module):
    """Project KG embeddings into the frozen LM hidden space.

    Input shape:
        [batch_size, kg_dim]

    Output shape:
        [batch_size, lm_dim]
    """

    def __init__(
        self,
        kg_dim: int = 32,
        lm_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kg_dim <= 0:
            raise ValueError("kg_dim must be > 0")

        if lm_dim <= 0:
            raise ValueError("lm_dim must be > 0")

        self.kg_dim = kg_dim
        self.lm_dim = lm_dim

        self.projection = nn.Sequential(
            nn.Linear(kg_dim, lm_dim),
            nn.LayerNorm(lm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, kg_vectors: torch.Tensor) -> torch.Tensor:
        """Project KG vectors into LM hidden space."""

        if kg_vectors.dim() != 2:
            raise ValueError(
                "kg_vectors must have shape [batch_size, kg_dim]"
            )

        if kg_vectors.size(-1) != self.kg_dim:
            raise ValueError(
                f"Expected KG dimension {self.kg_dim}, "
                f"but received {kg_vectors.size(-1)}"
            )

        return self.projection(kg_vectors)