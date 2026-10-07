"""Phase 4: second RGAT over text-enriched entities, followed by ComplEx."""
from models.kg_encoder_rgat import RGATEncoder
from models.kg_text_integration import KGTextIntegration
from models.scorer import ComplExScorer
import torch
from torch.utils.checkpoint import checkpoint


class KGTextRefinement(KGTextIntegration):
    """Small/full graph API; every refinement node needs an enriched vector.

    Structural edges use checkpoint-global IDs. Refinement edges use row
    indices in entity_ids. Relation edge types retain original global IDs
    (plus inverse offset), while scoring triples use local entity/relation rows.
    """

    def __init__(self, structural_model, lm_name='roberta-base',
                 projection_dropout=0.1, refinement_layers=2, refinement_heads=1):
        super().__init__(structural_model, lm_name, projection_dropout)
        self.refiner = RGATEncoder(
            dim=structural_model.entity_emb.embedding_dim,
            num_message_relations=2 * structural_model.num_relations,
            num_layers=refinement_layers, heads=refinement_heads, dropout=0.2,
        )

    def forward(self, edge_index, edge_type, entity_ids, relation_ids,
                entity_tokens, relation_tokens, refinement_edge_index,
                text_batch_size=None):
        if text_batch_size is None:
            entities, relations = super().forward(
                edge_index, edge_type, entity_ids, relation_ids,
                entity_tokens, relation_tokens,
            )
        else:
            if text_batch_size < 1:
                raise ValueError('text_batch_size must be positive')
            structural = self.structural.encode(edge_index, edge_type)
            def enrich(vectors, tokens):
                parts = []
                for start in range(0, len(vectors), text_batch_size):
                    stop = start + text_batch_size
                    inputs = (vectors[start:stop], tokens['input_ids'][start:stop],
                              tokens['attention_mask'][start:stop])
                    if self.training and torch.is_grad_enabled():
                        parts.append(checkpoint(self.bridge, *inputs, use_reentrant=False))
                    else:
                        parts.append(self.bridge(*inputs))
                return torch.cat(parts)
            entities = enrich(structural[entity_ids], entity_tokens)
            relations = enrich(self.structural.scorer.relation_emb(relation_ids), relation_tokens)
        return self.refiner(entities, refinement_edge_index, edge_type), relations

    @staticmethod
    def score_triples(entities, relations, triples):
        return ComplExScorer.score_vectors(
            entities[triples[:, 0]], relations[triples[:, 1]],
            entities[triples[:, 2]],
        )
