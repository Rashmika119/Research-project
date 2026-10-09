"""All-candidate ComplEx scores using the same explicit relations as training."""


class EnrichedScorer:
    def __init__(self, relations):
        self.relations = relations

    def score_all_tails(self, entities, h, ids):
        hr, hi = h.chunk(2, -1)
        rr, ri = self.relations[ids].chunk(2, -1)
        er, ei = entities.chunk(2, -1)
        return (hr * rr - hi * ri) @ er.t() + (hr * ri + hi * rr) @ ei.t()

    def score_all_heads(self, entities, ids, t):
        rr, ri = self.relations[ids].chunk(2, -1)
        tr, ti = t.chunk(2, -1)
        er, ei = entities.chunk(2, -1)
        return (rr * tr + ri * ti) @ er.t() + (rr * ti - ri * tr) @ ei.t()
