"""Filtered negative sampling: corrupt head or tail, reject anything that is
a known-true triple.

`known_true` should only ever be built from TRAINING triples (CLAUDE.md rule
#3 — validation/test labels must never leak into what training treats as
"true"). This is a different, narrower "known" set than the one
`evaluation.metrics.build_filter_index` builds from all splits combined for
ranking purposes — don't conflate the two.
"""

from __future__ import annotations

import random

import torch

IdTriple = tuple[int, int, int]


def sample_negatives(
    pos_triples: torch.Tensor,
    num_entities: int,
    known_true: set[IdTriple],
    num_negatives: int,
    rng: random.Random,
    max_attempts_per_negative: int = 100,
) -> torch.Tensor:
    """Returns a [batch, num_negatives, 3] long tensor of corrupted triples.

    For each positive triple, corrupts either the head or the tail (coin
    flip) with a random entity, resampling if the result happens to already
    be a known-true triple. `pos_triples` must be a CPU tensor (this
    function loops in plain Python) — move it there before calling.
    """
    batch = pos_triples.tolist()
    out = torch.empty((len(batch), num_negatives, 3), dtype=torch.long)

    for i, (h, r, t) in enumerate(batch):
        for k in range(num_negatives):
            for _ in range(max_attempts_per_negative):
                corrupt_head = rng.random() < 0.5
                e = rng.randrange(num_entities)
                hh, tt = (e, t) if corrupt_head else (h, e)
                if (hh, tt) == (h, t):
                    continue
                if (hh, r, tt) not in known_true:
                    out[i, k] = torch.tensor([hh, r, tt])
                    break
            else:
                raise RuntimeError(
                    "Negative sampler failed to find a valid corruption "
                    f"after {max_attempts_per_negative} attempts for triple "
                    f"{(h, r, t)} — num_entities may be too small relative "
                    "to how densely connected this entity is (more likely "
                    "on a very small toy subset than on the full dataset)."
                )
    return out
