"""Two unique RGAT layers with the Shared C gated computation path."""
import torch
from torch.nn import functional as F

from models.kg_only_baseline_rgat import KGOnlyBaselineRGAT
from models.kg_text_integration import KGTextIntegration
from models.scorer import ComplExScorer


class SharedStructuralRGAT(KGOnlyBaselineRGAT):
    """Own the only two graph layers; no aliases or duplicate registered modules.

    L1 always means RGATConv -> ReLU -> dropout, in both passes.
    L2 is the final RGATConv without an added activation. Repeated calls share
    weights, but draw fresh dropout masks during training.
    """
    def __init__(self, *args, arrangement, **kwargs):
        if arrangement != 'c' or kwargs.get('num_layers', 2) != 2:
            raise ValueError('Shared RGAT requires arrangement c and exactly two layers')
        super().__init__(*args, **kwargs)
        self.arrangement = arrangement

    def apply_layer(self, index, vectors, edge_index, edge_type):
        vectors = self.encoder.layers[index](vectors, edge_index, edge_type)
        return self.encoder.dropout(F.relu(vectors)) if index == 0 else vectors

    def encode(self, edge_index, edge_type):
        vectors = self.apply_layer(0, self.entity_emb.weight, edge_index, edge_type)
        vectors = self.apply_layer(1, vectors, edge_index, edge_type)
        return vectors

    def refine(self, vectors, edge_index, edge_type):
        vectors = self.apply_layer(0, vectors, edge_index, edge_type)
        return self.apply_layer(1, vectors, edge_index, edge_type)


class SharedRGATRefinement(KGTextIntegration):
    """Independent text fusion followed by shared refinement and residual addition."""
    def __init__(self, structural_model, *, refinement_gated, lm_name='roberta-base', lm_revision=None):
        if refinement_gated is not True:
            raise ValueError('Only gated refinement is supported')
        super().__init__(structural_model, lm_name=lm_name, soft_prompt=False, lm_revision=lm_revision)
        self.refinement_gated = refinement_gated
        self.entity_text_gate = torch.nn.Parameter(torch.tensor(.01))
        self.relation_text_gate = torch.nn.Parameter(torch.tensor(.01))
        self.refinement_gate = torch.nn.Parameter(torch.tensor(.01))

    def forward(self, edge_index, edge_type, entity_ids, relation_ids,
                entity_tokens, relation_tokens, refinement_edge_index,
                text_batch_size=None, entity_pooled=None, relation_pooled=None):
        if text_batch_size is not None and text_batch_size < 1:
            raise ValueError('text_batch_size must be positive')
        entities = self.structural.encode(edge_index, edge_type)[entity_ids]
        relations = self.structural.scorer.relation_emb(relation_ids)

        def project(vectors, tokens, pooled):
            size = text_batch_size or len(vectors)
            return torch.cat([
                self.bridge(pooled=pooled[start:start+size].to(vectors.device))
                if pooled is not None else self.bridge(
                    tokens['input_ids'][start:start+size].to(vectors.device),
                    tokens['attention_mask'][start:start+size].to(vectors.device))
                for start in range(0, len(vectors), size)])

        fused = entities + self.entity_text_gate.tanh() * project(entities, entity_tokens, entity_pooled)
        relations = relations + self.relation_text_gate.tanh() * project(relations, relation_tokens, relation_pooled)
        refined = self.structural.refine(fused, refinement_edge_index, edge_type)
        entities = fused + self.refinement_gate.tanh() * refined
        return entities, relations

    @staticmethod
    def score_triples(entities, relations, triples):
        return ComplExScorer.score_vectors(entities[triples[:, 0]], relations[triples[:, 1]], entities[triples[:, 2]])
