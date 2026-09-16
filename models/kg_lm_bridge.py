"""Complete Phase 2 KG -> LM -> KG semantic bridge.

Flow:

    KG vector
    [batch, kg_dim]
          ↓
    KG -> LM projection
          ↓
    soft prompt
    [batch, lm_dim]
          ↓
    Frozen language model + entity description
          ↓
    contextualized semantic representation
    [batch, lm_dim]
          ↓
    LM -> KG projection
          ↓
    semantic-enriched KG vector
    [batch, kg_dim]

The language model remains frozen. Only the projection modules are trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.kg_lm_projection import KGLMProjection
from models.frozen_lm import FrozenLM
from models.lm_kg_projection import LMKGProjection


class KGLMBridge(nn.Module):
    """Phase 2 semantic bridge between KG space and frozen LM space."""

    def __init__(
        self,
        kg_dim: int = 32,
        model_name: str = "bert-base-uncased",
        projection_dropout: float = 0.1,
    ):
        super().__init__()

        self.kg_dim = kg_dim

        # Frozen language model.
        self.lm = FrozenLM(
            model_name=model_name,
        )

        self.lm_dim = self.lm.hidden_size

        # Trainable KG -> LM mapping.
        self.kg_to_lm = KGLMProjection(
            kg_dim=kg_dim,
            lm_dim=self.lm_dim,
            dropout=projection_dropout,
        )

        # Trainable LM -> KG mapping.
        self.lm_to_kg = LMKGProjection(
            lm_dim=self.lm_dim,
            kg_dim=kg_dim,
            dropout=projection_dropout,
        )

    def forward(
        self,
        kg_vectors: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich KG vectors with textual semantics.

        Parameters
        ----------
        kg_vectors:
            Structural KG representations:

                [batch_size, kg_dim]

        input_ids:
            Tokenized entity descriptions:

                [batch_size, seq_len]

        attention_mask:
            Attention mask for the descriptions:

                [batch_size, seq_len]

        Returns
        -------
        torch.Tensor
            Semantic-enriched KG representations:

                [batch_size, kg_dim]
        """

        # 1. Move structural KG representation into LM hidden space.
        soft_prompt = self.kg_to_lm(
            kg_vectors
        )

        # 2. Combine the projected KG vector with entity text and let the
        # frozen LM contextualize it.
        semantic_lm_vector = self.lm(
            soft_prompt=soft_prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 3. Project the LM semantic representation back into KG space.
        semantic_kg_vector = self.lm_to_kg(
            semantic_lm_vector
        )

        return semantic_kg_vector