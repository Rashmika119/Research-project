"""Known-answer tests for average ties and filtered ranking."""
from types import SimpleNamespace
import torch
from evaluation.metrics import ranks_with_ties, evaluate_filtered, build_filter_index


def main():
    scores = torch.tensor([[.2, .2, .2, .2], [3., 2., 2., 1.], [3., 2., 1., 0.]])
    average, optimistic, ties = ranks_with_ties(scores, torch.tensor([.2, 2., 2.]))
    torch.testing.assert_close(average, torch.tensor([2.5, 2.5, 2.]))
    torch.testing.assert_close(optimistic, torch.tensor([1., 2., 2.]))
    assert ties.tolist() == [4, 2, 1]
    class EqualScorer:
        def score_all_tails(self, entities, h, ids):
            return torch.zeros(len(h), len(entities))
        def score_all_heads(self, entities, ids, t):
            return torch.zeros(len(t), len(entities))
    details = []
    metrics = evaluate_filtered(SimpleNamespace(scorer=EqualScorer()), torch.zeros(4, 2),
        [(0, 0, 1)], *build_filter_index([(0, 0, 1), (0, 0, 2)]), diagnostics=details)
    # Tail: one other known answer removed => 3 ties, rank 2.
    # Head: 4 ties, rank 2.5. MRR = (1/2 + 1/2.5)/2 = .45.
    assert abs(metrics['MRR'] - .45) < 1e-6
    assert metrics['Optimistic_MRR'] == 1. and metrics['Hits@1'] == 0.
    assert [d['tied_candidates_including_true'] for d in details] == [3, 4]
    print('Tie-aware ranking checks PASSED')


if __name__ == '__main__':
    main()
