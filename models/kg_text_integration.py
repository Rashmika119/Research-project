"""Phase 3: learned graph entities and relations through the semantic bridge."""
from torch import nn

from models.kg_lm_bridge import KGLMBridge


class KGTextIntegration(nn.Module):
    """Shares one frozen LM bridge for entities and relations; no second GNN yet."""

    def __init__(self, structural_model, lm_name="roberta-base", projection_dropout=0.1):
        super().__init__()
        self.structural = structural_model
        self.bridge = KGLMBridge(
            kg_dim=structural_model.entity_emb.embedding_dim,
            model_name=lm_name,
            projection_dropout=projection_dropout,
        )

    def forward(self, edge_index, edge_type, entity_ids, relation_ids,
                entity_tokens, relation_tokens):
        # Encode once and retain autograd through both learned vector sources.
        entity_repr = self.structural.encode(edge_index, edge_type)
        entities = self.bridge(
            entity_repr[entity_ids], entity_tokens["input_ids"],
            entity_tokens["attention_mask"],
        )
        relations = self.bridge(
            self.structural.scorer.relation_emb(relation_ids),
            relation_tokens["input_ids"], relation_tokens["attention_mask"],
        )
        return entities, relations
