import numpy as np
from scipy.optimize import linear_sum_assignment

from cps.metrics import finite_horizon_gain
from cps.spectra import spectral_sweep


def _assert_same_eigenvalue_multiset(actual, expected, *, atol=1e-10, rtol=1e-8):
    """Compare unordered spectra without relying on floating-point sort tie-breaks."""

    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    assert actual.shape == expected.shape
    costs = np.abs(actual[:, None] - expected[None, :])
    rows, cols = linear_sum_assignment(costs)
    assert np.allclose(actual[rows], expected[cols], atol=atol, rtol=rtol)


def test_two_by_two_matches_closed_form_set():
    a = np.array([[0.2, 0.7], [0.4, -0.1]], dtype=complex)
    phases = np.linspace(0.0, 2.0 * np.pi, 17)
    result = spectral_sweep(a, 0, 1, phases=phases, compute_kreiss=False)
    for phi, values in zip(phases, result.eigenvalues):
        discriminant = (a[0, 0] - a[1, 1]) ** 2 + 4 * a[1, 0] * a[0, 1] * np.exp(1j * phi)
        expected = np.array([
            (a[0, 0] + a[1, 1] + np.sqrt(discriminant)) / 2,
            (a[0, 0] + a[1, 1] - np.sqrt(discriminant)) / 2,
        ])
        _assert_same_eigenvalue_multiset(values, expected)


def test_acyclic_edge_has_no_spectral_motion():
    # Strictly upper-triangular edge changes no eigenvalue because it lies on no directed cycle.
    a = np.array([[1.0, 2.0, 0.0], [0.0, 2.0, 3.0], [0.0, 0.0, 3.0]])
    result = spectral_sweep(a, 0, 1, compute_kreiss=False)
    assert np.max(np.abs(result.eigenvalues - result.eigenvalues[0])) < 1e-10


def test_nonnormal_matrix_can_have_transient_gain_above_one():
    a = np.array([[0.8, 4.0], [0.0, 0.8]])
    assert finite_horizon_gain(a, horizon=8) > 1.0


def test_diagonal_unitary_similarity_preserves_spectrum():
    a = np.array(
        [[0.5, 0.7, 0.0], [0.0, 0.6, 0.8], [0.4, 0.0, 0.7]], dtype=complex
    )
    phases = np.array([0.2, -0.4, 0.9])
    d = np.diag(np.exp(1j * phases))
    b = d @ a @ np.linalg.inv(d)
    _assert_same_eigenvalue_multiset(np.linalg.eigvals(a), np.linalg.eigvals(b))


def test_trace_moment_harmonic_matches_closed_walk_count():
    from cps.moments import phase_moment_spectrum

    # A 2-cycle. tr(A^2) contains 2*a01*a10 and therefore one first harmonic.
    a = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=complex)
    spectrum = phase_moment_spectrum(a, 0, 1, max_order=2)
    coeffs = spectrum[2]
    assert np.isclose(coeffs[1], 12.0, atol=1e-10)
    assert np.max(np.abs(coeffs[2:])) < 1e-9
