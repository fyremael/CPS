import pandas as pd

from cps.pythia.variance import decompose_step_seed_variance


def test_variance_decomposition_nonnegative():
    frame = pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "seed": [1, 2, 1, 2],
            "value": [0.0, 0.1, 1.0, 1.1],
        }
    )
    result = decompose_step_seed_variance(frame, "value")
    assert result.total_variance > 0
    assert result.between_step_variance > result.between_seed_variance
    assert result.residual_variance >= 0
