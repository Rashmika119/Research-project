"""Train scratch architecture variants; historical checkpoint mode is explicit."""
import sys

# Retain the public import used by historical checkpoint reevaluation.
from evaluation.enriched_scorer import EnrichedScorer


def main():
    if '--legacy-pretrained' in sys.argv:
        sys.argv.remove('--legacy-pretrained')
        from training.legacy_phase5 import main as legacy_main
        legacy_main()
    else:
        from training.research_cli import main as scratch_main
        scratch_main()


if __name__ == '__main__':
    main()
