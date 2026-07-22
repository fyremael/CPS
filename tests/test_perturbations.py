import numpy as np

from cps.perturbations import phase_perturbation_norm, phase_rotate_entry, real_pair_rotation


def test_phase_rotation_preserves_magnitude():
    a = np.array([[1.0, -2.0], [0.5, 3.0]])
    b = phase_rotate_entry(a, 0, 1, np.pi / 3)
    assert np.isclose(abs(b[0, 1]), abs(a[0, 1]))
    assert np.allclose(b[1, :], a[1, :])


def test_phase_perturbation_norm_matches_matrix_norm():
    a = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex)
    phi = 0.7
    b = phase_rotate_entry(a, 0, 1, phi)
    assert np.isclose(np.linalg.norm(b - a, 2), phase_perturbation_norm(a[0, 1], phi))


def test_real_pair_rotation_preserves_joint_norm():
    a = np.array([[0.0, 3.0], [4.0, 0.0]])
    b = real_pair_rotation(a, (0, 1), (1, 0), 0.6)
    assert np.isclose(a[0, 1] ** 2 + a[1, 0] ** 2, b[0, 1].real ** 2 + b[1, 0].real ** 2)


def test_block_phase_preserves_singular_values():
    from cps.perturbations import rotate_block_svd_phase

    a = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=float
    )
    before = np.linalg.svd(a[1:3, 1:3], compute_uv=False)
    b = rotate_block_svd_phase(a, slice(1, 3), slice(1, 3), np.array([0.4, -0.2]))
    after = np.linalg.svd(b[1:3, 1:3], compute_uv=False)
    assert np.allclose(before, after)
