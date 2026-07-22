import numpy as np
import torch

from cps.pythia.basis import BasisVector
from cps.pythia.reduced_operator import project_jacobian


def test_project_jacobian_recovers_dense_operator():
    matrix = torch.tensor([[0.8, 0.2], [-0.1, 0.9]], dtype=torch.float64)
    basis = (
        BasisVector("e0", torch.tensor([1.0, 0.0], dtype=torch.float64), "x", "test"),
        BasisVector("e1", torch.tensor([0.0, 1.0], dtype=torch.float64), "y", "test"),
    )
    reduced, diagnostics = project_jacobian(lambda x: matrix @ x, basis)
    assert np.allclose(reduced, matrix.numpy())
    assert diagnostics.maximum_closure_residual < 1e-12
