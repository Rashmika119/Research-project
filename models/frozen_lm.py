"""Frozen RoBERTa with independent masked-mean text pooling."""
from __future__ import annotations

import torch
import torch.nn as nn
from models.pretrained import load_pretrained_model


class FrozenLM(nn.Module):
    """Frozen RoBERTa encoder shared by the two text schedules."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        revision: str | None = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.lm = load_pretrained_model(model_name, revision=revision)

        # Freeze all RoBERTa parameters.
        for param in self.lm.parameters():
            param.requires_grad = False

        # Keep RoBERT in evaluation mode so its dropout layers remain disabled.
        self.lm.eval()

        self.hidden_size = self.lm.config.hidden_size

    def train(self, mode: bool = True):
        """Allow the wrapper to enter train mode while RoBERT stays in eval mode."""

        super().train(mode)

        # `model.train()` would normally switch every child module,
        # including RoBERT, into training mode. We deliberately keep the
        # frozen LM in evaluation mode.
        self.lm.eval()

        return self

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        """Independent text encoding: no structural inputs or prompt tokens.

        Mean pooling includes every unmasked token (including tokenizer special
        tokens), and excludes padding. Only these frozen outputs may be cached.
        """
        if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
            raise ValueError('Text IDs and attention mask must be matching matrices')
        if (attention_mask.sum(1) == 0).any():
            raise ValueError('Cannot pool an entirely masked description')
        self.lm.eval()
        hidden = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                         return_dict=True).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1)
