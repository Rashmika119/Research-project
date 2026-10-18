"""Known-answer ComplEx and filtered-ranking checks; no optimizer steps."""
import unittest


class ScoringTests(unittest.TestCase):
    def test_complex_algebra_candidate_scores_and_gradients(self):
        from run_complex_scorer_check import main
        main()

    def test_exact_ties_and_filtered_ranking(self):
        from run_tie_ranking_check import main
        main()
