"""Frozen language-model wrapper with KG soft-prompt injection.

A projected KG vector is inserted as an additional continuous prompt token
before the token embeddings of a KG item description.

Flow:

    projected KG vector [batch, lm_dim]
                +
    description token embeddings [batch, seq_len, lm_dim]
                ↓
    [soft prompt | description tokens]
                ↓
           frozen RoBERT
                ↓
    contextualized soft-prompt representation [batch, lm_dim]

RoBERTa parameters remain frozen, but gradients are still allowed to flow
through RoBERT operations back to the trainable KG -> LM projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenLM(nn.Module):
    """Frozen RoBERT encoder used by the Phase 2 KG-LM bridge."""

    def __init__(
        self,
        model_name: str = "roberta-base",
    ):
        super().__init__()

        self.model_name = model_name
        self.lm = AutoModel.from_pretrained(model_name)

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

    def forward(
        self,
        soft_prompt: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject one KG soft-prompt token before the description tokens.

        Parameters
        ----------
        soft_prompt:
            Projected KG vectors:

                [batch_size, hidden_size]

        input_ids:
            Tokenized KG item descriptions:

                [batch_size, seq_len]

        attention_mask:
            LM attention mask:

                [batch_size, seq_len]

        Returns
        -------
        torch.Tensor
            Contextualized representation at the soft-prompt position:

                [batch_size, hidden_size]
        """

        if soft_prompt.dim() != 2:
            raise ValueError(
                "soft_prompt must have shape [batch_size, hidden_size]"
            )

        if soft_prompt.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected soft-prompt dimension {self.hidden_size}, "
                f"but received {soft_prompt.size(-1)}"
            )

        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, seq_len]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids"
            )

        if soft_prompt.size(0) != input_ids.size(0):
            raise ValueError(
                "soft_prompt and input_ids must have the same batch size"
            )

        # Convert normal text token IDs into RoBERT token embeddings.
        token_embeddings = self.lm.get_input_embeddings()(input_ids)

        # Convert:
        #
        #   [batch, hidden]
        #
        # into:
        #
        #   [batch, 1, hidden]
        #
        # so it behaves like one extra continuous prompt token.
        prompt_token = soft_prompt.unsqueeze(1)

        # Prepend the KG-derived soft prompt to the description tokens.
        inputs_embeds = torch.cat(
            [prompt_token, token_embeddings],
            dim=1,
        )

        # The new prompt token must also be visible to RoBERTa attention.
        prompt_mask = torch.ones(
            (
                attention_mask.size(0),
                1,
            ),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        extended_attention_mask = torch.cat(
            [prompt_mask, attention_mask],
            dim=1,
        )

        # IMPORTANT:
        # Do NOT wrap this in torch.no_grad().
        #
        # LM parameters are frozen, but the computation graph must remain
        # available so gradients can flow back into the trainable KG -> LM
        # projection through the soft prompt.
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )

        # Position 0 is our KG-derived soft-prompt token.
        contextualized_prompt = outputs.last_hidden_state[:, 0, :]

        return contextualized_prompt