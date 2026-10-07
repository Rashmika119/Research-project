"""Phase 4: second RGAT over text-enriched entities, followed by ComplEx."""
from models.kg_encoder_rgat import RGATEncoder
from models.kg_text_integration import KGTextIntegration
from models.scorer import ComplExScorer


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
                entity_tokens, relation_tokens, refinement_edge_index):
        entities, relations = super().forward(
            edge_index, edge_type, entity_ids, relation_ids,
            entity_tokens, relation_tokens,
        )
        return self.refiner(entities, refinement_edge_index, edge_type), relations

    @staticmethod
    def score_triples(entities, relations, triples):
        return ComplExScorer.score_vectors(
            entities[triples[:, 0]], relations[triples[:, 1]],
            entities[triples[:, 2]],
        )
