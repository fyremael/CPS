import torch

from cps.pythia.basis import build_semantic_basis
from cps.pythia.state_layout import StateLayout


def test_semantic_basis_is_orthonormal():
    params = {"a": torch.arange(1.0, 5.0).reshape(2, 2)}
    layout = StateLayout.from_parameters(params)
    base = torch.zeros(layout.state_numel)
    base[layout.tensor("a").theta] = params["a"].reshape(-1)
    next_state = base.clone()
    next_state[layout.tensor("a").theta] -= 0.1
    basis = build_semantic_basis(
        layout,
        base,
        next_state,
        components=("theta", "update", "random"),
        random_vectors_per_block=2,
        rank=4,
        seed=1,
        tolerance=1e-8,
    )
    matrix = torch.stack([item.vector for item in basis])
    assert torch.allclose(matrix @ matrix.T, torch.eye(len(basis)), atol=1e-5)
