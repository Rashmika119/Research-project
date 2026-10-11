"""Shared structural/independent-text module layout for checkpoint compatibility."""
from torch import nn

from models.independent_text_bridge import IndependentTextBridge


class KGTextIntegration(nn.Module):
    """Initialize shared modules in their historical order and under the same keys."""

    def __init__(self, structural_model, lm_name="roberta-base", projection_dropout=0.1,
                 soft_prompt=False, lm_revision=None):
        super().__init__()
        if soft_prompt:
            raise ValueError('KG-conditioned soft prompts have been retired')
        self.structural = structural_model
        self.soft_prompt = soft_prompt
        self.bridge = IndependentTextBridge(
            kg_dim=structural_model.entity_emb.embedding_dim,
            model_name=lm_name,
            projection_dropout=projection_dropout,
            lm_revision=lm_revision,
        )
