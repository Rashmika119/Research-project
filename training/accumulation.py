"""Micro-batched ComplEx scoring with one encoder backward per effective batch."""
import torch

from models.scorer import ComplExScorer
from training.losses import bce_loss


def backward_effective_batch(encode, positives, negatives, micro_batch_size, device):
    """Accumulate example-weighted gradients without retaining scoring graphs.

    Inputs stay on CPU until each scoring micro-batch. Encode once, accumulate
    gradients on detached entity/relation leaves, then apply their vector-Jacobian
    product to the original graph. This keeps the effective batch's single dropout
    realization and avoids eight full-graph encodings/backwards. Encoder activations
    still remain resident until the final backward. No optimizer step occurs here.
    """
    count = len(positives)
    if count < 1 or micro_batch_size < 1 or len(negatives) != count:
        raise ValueError('Invalid effective/micro batch sizes')
    entities, relations = encode()
    micro = min(micro_batch_size, count)
    accumulate = micro < count
    e = entities.detach().requires_grad_(True) if accumulate else entities
    r = relations.detach().requires_grad_(True) if accumulate else relations
    total_loss = 0.
    for start in range(0, count, micro):
        positive = positives[start:start+micro].to(device)
        negative = negatives[start:start+micro].to(device)
        def score(triples):
            return ComplExScorer.score_vectors(e[triples[:,0]], r[triples[:,1]], e[triples[:,2]])
        loss = bce_loss(score(positive), score(negative.reshape(-1,3)).reshape(len(positive), -1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        weight = len(positive) / count
        (loss * weight).backward()
        total_loss += loss.detach().item() * weight
        del positive, negative, loss
    if accumulate:
        torch.autograd.backward((entities, relations), (e.grad, r.grad))
    return total_loss
