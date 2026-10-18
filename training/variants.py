"""Scratch model factory and explicit structural warmup accounting."""
from training.model_factory import build_model


ARCHITECTURES = ('shared-c-gated',)
VARIANTS = ARCHITECTURES
ALL_ARCHITECTURES = ARCHITECTURES
CONFIGURATIONS = {'baseline': ('baseline', False),
                  'shared-c-gated': ('shared-c-gated', True)}


def independent_text(variant):
    return variant == 'shared-c-gated'


MODEL_CONFIG = {'encoder_type': 'rgat', 'scorer_type': 'complex', 'dim': 32,
                'num_layers': 2, 'heads': 1, 'dropout': 0.2, 'num_bases': None}


def warmup_epochs(variant, total_epochs, requested):
    if total_epochs < 1 or requested < 0:
        raise ValueError('Epoch counts must be positive (warmup may be zero)')
    if variant not in (*ALL_ARCHITECTURES, 'baseline'):
        raise ValueError('Unknown architecture: ' + variant)
    actual = 0 if variant == 'baseline' else requested
    if actual >= total_epochs:
        raise ValueError('Warmup is INCLUDED in epochs; leave at least one integrated epoch')
    return actual


def configuration_id(architecture, warmup):
    if architecture not in CONFIGURATIONS:
        raise ValueError('Unknown architecture: ' + architecture)
    return architecture


def configuration_settings(identifier, requested_warmup=10):
    architecture, enabled = CONFIGURATIONS[identifier]
    if requested_warmup < 0 or (enabled and requested_warmup < 1):
        raise ValueError('Warmup-enabled configurations require positive warmup epochs')
    return architecture, requested_warmup if enabled else 0


def build_scratch_model(variant, num_entities, num_relations,
                        model_config=None, lm_name='roberta-base', lm_revision=None):
    """No checkpoint or embedding-cache parameter: KG weights always start fresh."""
    if variant not in (*ALL_ARCHITECTURES, 'baseline'):
        raise ValueError('Unknown variant: ' + variant)
    if variant == 'shared-c-gated':
        from models.shared_rgat import SharedStructuralRGAT, SharedRGATRefinement
        settings = dict(model_config or MODEL_CONFIG)
        if settings.pop('encoder_type', 'rgat') != 'rgat' or settings.get('scorer_type') != 'complex':
            raise ValueError('Shared study requires RGAT + ComplEx')
        arrangement, gated = 'c', True
        structural = SharedStructuralRGAT(num_entities, num_relations, arrangement=arrangement, **settings)
        return SharedRGATRefinement(structural, refinement_gated=gated, lm_name=lm_name, lm_revision=lm_revision)
    return build_model(model_config or MODEL_CONFIG, num_entities, num_relations)
