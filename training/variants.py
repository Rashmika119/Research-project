"""Scratch model factory and explicit structural warmup accounting."""
from models.kg_text_refinement import KGTextRefinement
from training.model_factory import build_model


ARCHITECTURES = ('residual', 'residual-no-softprompt', 'no-refinement')
VARIANTS = ARCHITECTURES  # compatibility name for architecture-oriented callers
# Architecture and warmup are independent; no duplicate model classes.
CONFIGURATIONS = {
    'residual-warmup': ('residual', True),
    'residual-no-warmup': ('residual', False),
    'residual-no-softprompt-warmup': ('residual-no-softprompt', True),
    'residual-no-softprompt-no-warmup': ('residual-no-softprompt', False),
}
MODEL_CONFIG = {'encoder_type': 'rgat', 'scorer_type': 'complex', 'dim': 32,
                'num_layers': 2, 'heads': 1, 'dropout': 0.2, 'num_bases': None}


def warmup_epochs(variant, total_epochs, requested):
    if total_epochs < 1 or requested < 0:
        raise ValueError('Epoch counts must be positive (warmup may be zero)')
    if variant not in (*ARCHITECTURES, 'baseline'):
        raise ValueError('Unknown architecture: ' + variant)
    actual = 0 if variant == 'baseline' else requested
    if actual >= total_epochs:
        raise ValueError('Warmup is INCLUDED in epochs; leave at least one integrated epoch')
    return actual


def configuration_id(architecture, warmup):
    if architecture == 'baseline':
        return 'baseline'
    return architecture + ('-warmup' if warmup > 0 else '-no-warmup')


def configuration_settings(identifier, requested_warmup=5):
    architecture, enabled = CONFIGURATIONS[identifier]
    if requested_warmup < 0 or (enabled and requested_warmup < 1):
        raise ValueError('Warmup-enabled configurations require positive warmup epochs')
    return architecture, requested_warmup if enabled else 0


def build_scratch_model(variant, num_entities, num_relations,
                        model_config=None, lm_name='roberta-base', lm_revision=None):
    """No checkpoint or embedding-cache parameter: KG weights always start fresh."""
    if variant not in (*VARIANTS, 'baseline'):
        raise ValueError('Unknown variant: ' + variant)
    structural = build_model(model_config or MODEL_CONFIG, num_entities, num_relations)
    if variant == 'baseline':
        return structural
    return KGTextRefinement(
        structural, lm_name=lm_name,
        residual=True,
        use_refinement=variant != 'no-refinement',
        soft_prompt=variant != 'residual-no-softprompt',
        lm_revision=lm_revision,
    )
