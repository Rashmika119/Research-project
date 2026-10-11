"""Gated independent-text fusion and residual RGAT refinement for both schedules."""
from models.kg_encoder_rgat import RGATEncoder
from models.kg_text_integration import KGTextIntegration
from models.scorer import ComplExScorer
import torch


class KGTextRefinement(KGTextIntegration):
    """Small/full graph API; every refinement node needs an enriched vector.

    Structural edges use checkpoint-global IDs. Refinement edges use row
    indices in entity_ids. Relation edge types retain original global IDs
    (plus inverse offset), while scoring triples use local entity/relation rows.
    """

    def __init__(self, structural_model, lm_name='roberta-base',
                 projection_dropout=0.1, refinement_layers=2, refinement_heads=1,
                 residual=True, use_refinement=True, soft_prompt=False, lm_revision=None):
        if not residual or not use_refinement or soft_prompt:
            raise ValueError('Only residual no-softprompt with refinement is supported')
        super().__init__(structural_model, lm_name, projection_dropout, soft_prompt, lm_revision)
        self.residual = residual
        self.use_refinement = use_refinement
        # Near-structural scratch initialization; retain parameter registration order.
        self.entity_text_gate = torch.nn.Parameter(torch.tensor(0.01))
        self.relation_text_gate = torch.nn.Parameter(torch.tensor(0.01))
        self.refinement_gate = torch.nn.Parameter(torch.tensor(0.01))
        self.refiner = RGATEncoder(
            dim=structural_model.entity_emb.embedding_dim,
            num_message_relations=2 * structural_model.num_relations,
            num_layers=refinement_layers, heads=refinement_heads, dropout=0.2,
        )

    def forward(self, edge_index, edge_type, entity_ids, relation_ids,
                entity_tokens, relation_tokens, refinement_edge_index,
                text_batch_size=None, entity_pooled=None, relation_pooled=None):
        if text_batch_size is not None and text_batch_size < 1:
            raise ValueError('text_batch_size must be positive')
        structural = self.structural.encode(edge_index, edge_type)
        base_entities = structural[entity_ids]
        base_relations = self.structural.scorer.relation_emb(relation_ids)
        def enrich(vectors, tokens, pooled):
            batch_size = text_batch_size or len(vectors)
            parts = []
            for start in range(0, len(vectors), batch_size):
                stop = start + batch_size
                if pooled is not None:
                    parts.append(self.bridge(pooled=pooled[start:stop].to(vectors.device)))
                else:
                    parts.append(self.bridge(tokens['input_ids'][start:stop].to(vectors.device),
                                             tokens['attention_mask'][start:stop].to(vectors.device)))
            return torch.cat(parts)
        entities = enrich(base_entities, entity_tokens, entity_pooled)
        relations = enrich(base_relations, relation_tokens, relation_pooled)
        entities = base_entities + self.entity_text_gate.tanh() * entities
        relations = base_relations + self.relation_text_gate.tanh() * relations
        refined = self.refiner(entities, refinement_edge_index, edge_type)
        entities = entities + self.refinement_gate.tanh() * refined
        return entities, relations

    @staticmethod
    def score_triples(entities, relations, triples):
        return ComplExScorer.score_vectors(
            entities[triples[:, 0]], relations[triples[:, 1]],
            entities[triples[:, 2]],
        )
