from types import SimpleNamespace

import torch

from cps.pythia.functional_adamw import AdamWHyperparameters, FunctionalAdamWProbe


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.projection = torch.nn.Linear(4, 11, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embedding(input_ids)
        logits = self.projection(hidden)
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return SimpleNamespace(logits=logits, loss=loss)


def test_functional_map_and_jvp_match_finite_difference():
    torch.manual_seed(0)
    model = TinyLM().double()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    }
    hyper = AdamWHyperparameters(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.01,
        step=1,
    )
    probe = FunctionalAdamWProbe(model, batch, ("projection.weight",), hyper)
    state = probe.encode_initial_state().requires_grad_(True)
    direction = torch.randn_like(state)
    direction = direction / direction.norm()
    autodiff = probe.jvp(state, direction, scaled=False)
    step = 1e-5
    plus = probe.map(state + step * direction)
    minus = probe.map(state - step * direction)
    finite = (plus - minus) / (2 * step)
    assert torch.allclose(autodiff, finite, atol=2e-4, rtol=2e-3)
