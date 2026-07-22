import numpy as np

from cps.families import singular_channel_family, sweep_matrix_family


def test_singular_channel_family_preserves_block_singular_values():
    matrix = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.complex128)
    family, metadata = singular_channel_family(matrix, [0], [1], channel=0)
    phases = np.linspace(0, 2 * np.pi, 9)
    baseline = np.linalg.svd(matrix[np.ix_([0], [1])], compute_uv=False)
    for phase in phases:
        current = family(float(phase))
        values = np.linalg.svd(current[np.ix_([0], [1])], compute_uv=False)
        assert np.allclose(values, baseline)
    result = sweep_matrix_family(
        family,
        phases,
        family_name="test",
        finite_horizon=3,
        compute_kreiss=False,
        metadata=metadata,
    )
    assert result.eigenvalues.shape == (9, 2)
    assert result.metadata["channel"] == 0
