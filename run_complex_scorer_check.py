"""Verify ComplEx algebra, candidate ranking, directionality and gradients."""
import torch

from models.scorer import ComplExScorer


def main():
    torch.manual_seed(0)
    scorer = ComplExScorer(3, 8)
    entities = torch.randn(5, 8, requires_grad=True)
    heads = entities[[0, 1]]
    tails = entities[[2, 3]]
    relations = torch.tensor([0, 1])
    h = torch.complex(*heads.chunk(2, -1))
    r = torch.complex(*scorer.relation_emb(relations).chunk(2, -1))
    t = torch.complex(*tails.chunk(2, -1))
    expected = (h * r * t.conj()).sum(-1).real
    actual = scorer.score(heads, relations, tails)
    torch.testing.assert_close(actual, expected)
    all_tails = scorer.score_all_tails(entities, heads, relations)
    all_heads = scorer.score_all_heads(entities, relations, tails)
    for candidate in range(len(entities)):
        values = entities[candidate].expand(2, -1)
        torch.testing.assert_close(all_tails[:, candidate], scorer.score(heads, relations, values))
        torch.testing.assert_close(all_heads[:, candidate], scorer.score(values, relations, tails))
    directed = ComplExScorer(1, 2)
    with torch.no_grad():
        directed.relation_emb.weight.copy_(torch.tensor([[0., 1.]]))
    left, right, rel = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), torch.tensor([0])
    assert directed.score(left, rel, right).item() == 1
    assert directed.score(right, rel, left).item() == -1
    actual.sum().backward()
    for grad in (entities.grad, scorer.relation_emb.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for dim in (0, 3):
        try:
            ComplExScorer(1, dim)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid dimension was accepted")
    print("ComplEx scorer checks PASSED")


if __name__ == "__main__":
    main()
