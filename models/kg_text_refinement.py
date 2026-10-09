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
                 projection_dropout=0.1, refinement_layers=2, refinement_heads=1,
                 residual=False, use_refinement=True, soft_prompt=True, lm_revision=None):
        super().__init__(structural_model, lm_name, projection_dropout, soft_prompt, lm_revision)
        self.residual = residual
        self.use_refinement = use_refinement
        if residual:
            # Start near the pretrained graph model while allowing gradients
            # through all new components from the first step.
            self.entity_text_gate = torch.nn.Parameter(torch.tensor(0.01))
            self.relation_text_gate = torch.nn.Parameter(torch.tensor(0.01))
            if use_refinement:
                self.refinement_gate = torch.nn.Parameter(torch.tensor(0.01))
        self.refiner = RGATEncoder(
            dim=structural_model.entity_emb.embedding_dim,
            num_message_relations=2 * structural_model.num_relations,
            num_layers=refinement_layers, heads=refinement_heads, dropout=0.2,
        ) if use_refinement else None

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
                if not self.soft_prompt:
                    if pooled is not None:
                        parts.append(self.bridge(pooled=pooled[start:stop].to(vectors.device)))
                    else:
                        parts.append(self.bridge(tokens['input_ids'][start:stop].to(vectors.device),
                                                 tokens['attention_mask'][start:stop].to(vectors.device)))
                    continue
                if pooled is not None:
                    raise ValueError('Soft-prompt outputs depend on KG weights and cannot be cached')
                inputs = (vectors[start:stop], tokens['input_ids'][start:stop],
                          tokens['attention_mask'][start:stop])
                inputs = tuple(value.to(vectors.device) for value in inputs)
                if text_batch_size is not None and self.training and torch.is_grad_enabled():
                    parts.append(checkpoint(self.bridge, *inputs, use_reentrant=False))
                else:
                    parts.append(self.bridge(*inputs))
            return torch.cat(parts)
        entities = enrich(base_entities, entity_tokens, entity_pooled)
        relations = enrich(base_relations, relation_tokens, relation_pooled)
        if self.residual:
            entities = base_entities + self.entity_text_gate.tanh() * entities
            relations = base_relations + self.relation_text_gate.tanh() * relations
        if self.refiner is not None:
            refined = self.refiner(entities, refinement_edge_index, edge_type)
            entities = entities + self.refinement_gate.tanh() * refined if self.residual else refined
        return entities, relations

    @staticmethod
    def score_triples(entities, relations, triples):
        return ComplExScorer.score_vectors(
            entities[triples[:, 0]], relations[triples[:, 1]],
            entities[triples[:, 2]],
        )
