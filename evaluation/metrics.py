"""Filtered link-prediction evaluation: MRR, Hits@1, Hits@3, Hits@10.

Filtering removes OTHER known-true candidates from the ranking (for both
head and tail prediction) before computing the rank of the correct answer —
the standard KGC evaluation protocol. `build_filter_index` should be built
from every split combined (train+valid+test) — this is a different,
broader "known" set than `training.negative_sampling`'s train-only one, and
serves a different purpose: it never touches training, only ranking.
"""

from __future__ import annotations

from collections import defaultdict

import torch

IdTriple = tuple[int, int, int]


def ranks_with_ties(scores, true_scores):
    """Average rank of exact ties; true candidate is included in the tie count."""
    higher = (scores > true_scores[:, None]).sum(1)
    equal = (scores == true_scores[:, None]).sum(1)
    optimistic = 1 + higher
    return optimistic.float() + (equal.float() - 1) / 2, optimistic.float(), equal


def build_filter_index(
    all_triples: list[IdTriple],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    """Returns (hr_to_tails, rt_to_heads) built from every split combined —
    used only to filter candidates out of a ranking, never as training
    signal.
    """
    hr_to_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    rt_to_heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in all_triples:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)
    return hr_to_tails, rt_to_heads


@torch.no_grad()
def evaluate_filtered(
    model,
    entity_repr: torch.Tensor,
    eval_triples: list[IdTriple],
    hr_to_tails: dict[tuple[int, int], set[int]],
    rt_to_heads: dict[tuple[int, int], set[int]],
    batch_size: int = 64,
    diagnostics: list | None = None,
) -> dict[str, float]:
    """Computes filtered MRR/Hits@K for tail and head prediction, averaged
    together into one set of numbers."""
    device = entity_repr.device
    triples_t = torch.tensor(eval_triples, dtype=torch.long, device=device)

    if not eval_triples or batch_size < 1:
        raise ValueError('Evaluation needs nonempty triples and positive batch_size')
    ranks, optimistic_ranks, ties = [], [], []
    def record(scores, true_scores, batch, direction):
        if not torch.isfinite(true_scores).all():
            raise ValueError('Non-finite true candidate scores')
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise ValueError('Non-finite candidate scores')
        rank, optimistic, equal = ranks_with_ties(scores, true_scores)
        ranks.append(rank)
        optimistic_ranks.append(optimistic)
        ties.append(equal)
        if diagnostics is not None:
            for i, triple in enumerate(batch.tolist()):
                finite = scores[i][torch.isfinite(scores[i])]
                diagnostics.append({'triple': triple, 'direction': direction,
                    'true_score': float(true_scores[i]), 'higher_candidates': int(optimistic[i] - 1),
                    'tied_candidates_including_true': int(equal[i]),
                    'remaining_candidates': len(finite), 'average_rank': float(rank[i]),
                    'optimistic_rank': float(optimistic[i]),
                    'score_min': float(finite.min()), 'score_max': float(finite.max())})
    for start in range(0, len(triples_t), batch_size):
        batch = triples_t[start : start + batch_size]
        h_ids, r_ids, t_ids = batch[:, 0], batch[:, 1], batch[:, 2]
        h = entity_repr[h_ids]
        t = entity_repr[t_ids]
        row_ids = torch.arange(len(batch), device=device)

        # --- Tail prediction: (h, r, ?) ---
        tail_scores = model.scorer.score_all_tails(entity_repr, h, r_ids)
        true_tail_scores = tail_scores[row_ids, t_ids].clone()
        for i, (hh, rr, tt) in enumerate(batch.tolist()):
            known = hr_to_tails.get((hh, rr), set())
            if known:
                idx = torch.tensor(list(known), dtype=torch.long, device=device)
                tail_scores[i, idx] = float("-inf")
            tail_scores[i, tt] = true_tail_scores[i]
        record(tail_scores, true_tail_scores, batch, 'tail')

        # --- Head prediction: (?, r, t) ---
        head_scores = model.scorer.score_all_heads(entity_repr, r_ids, t)
        true_head_scores = head_scores[row_ids, h_ids].clone()
        for i, (hh, rr, tt) in enumerate(batch.tolist()):
            known = rt_to_heads.get((rr, tt), set())
            if known:
                idx = torch.tensor(list(known), dtype=torch.long, device=device)
                head_scores[i, idx] = float("-inf")
            head_scores[i, hh] = true_head_scores[i]
        record(head_scores, true_head_scores, batch, 'head')

    all_ranks = torch.cat(ranks)
    all_ties = torch.cat(ties)
    return {
        "MRR": float((1.0 / all_ranks).mean()),
        "Hits@1": float((all_ranks <= 1).float().mean()),
        "Hits@3": float((all_ranks <= 3).float().mean()),
        "Hits@10": float((all_ranks <= 10).float().mean()),
        "Optimistic_MRR": float((1 / torch.cat(optimistic_ranks)).mean()),
        "Tie_query_fraction": float((all_ties > 1).float().mean()),
        "Mean_tied_candidates": float(all_ties.float().mean()),
        "Max_tied_candidates": int(all_ties.max()),
    }
