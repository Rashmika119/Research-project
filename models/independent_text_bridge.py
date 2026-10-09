"""Frozen description encoding and a trainable LM-to-KG projection."""
from torch import nn

from models.frozen_lm import FrozenLM
from models.lm_kg_projection import LMKGProjection


class IndependentTextBridge(nn.Module):
    """No KG-to-LM projection exists in this architecture."""

    def __init__(self, kg_dim=32, model_name='roberta-base', projection_dropout=0.1,
                 lm_revision=None):
        super().__init__()
        self.lm = FrozenLM(model_name, revision=lm_revision)
        self.lm_dim = self.lm.hidden_size
        self.lm_to_kg = LMKGProjection(self.lm_dim, kg_dim, projection_dropout)

    def forward(self, input_ids=None, attention_mask=None, pooled=None):
        if pooled is None:
            pooled = self.lm.encode_text(input_ids, attention_mask)
        # Never cache this projection: its parameters change at each update.
        return self.lm_to_kg(pooled.detach())
