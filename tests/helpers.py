"""Offline fixtures, never research measurements."""
from types import SimpleNamespace
import torch
from torch import nn
from preprocessing.dataset import load_synthetic_toy_graph
from training.research_pipeline import ExperimentConfig
from training.variants import MODEL_CONFIG

class TinyFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, _commit_hash='offline-test-revision')
        with torch.random.fork_rng():
            torch.manual_seed(987)
            self.embedding = nn.Embedding(64, 8)
            self.linear = nn.Linear(8, 8)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        self.calls.append({'input_ids': input_ids is not None,
                           'inputs_embeds': inputs_embeds is not None,
                           'length': attention_mask.shape[1]})
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        context = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.tanh(self.linear(x) + context))

def fake_lm(*args, **kwargs):
    return TinyFrozenLM()

def make_tokens(count):
    ids = (torch.arange(count * 5).reshape(count, 5) % 60) + 1
    mask = torch.ones_like(ids)
    mask[:, -2:] = 0
    return {'input_ids': ids, 'attention_mask': mask}

def tiny_data():
    return load_synthetic_toy_graph(num_entities=12, num_relations=3,
                                    num_train=35, num_valid=5, num_test=5, seed=3)

def small_config(**kwargs):
    return ExperimentConfig(max_entities=0, epochs=3, warmup=1,
                            model={**MODEL_CONFIG, 'dim': 8}, lm_name='offline-test', eval_every=1,
                            selection_policy='scheduled_only', **kwargs)
