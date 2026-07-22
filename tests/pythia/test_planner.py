import numpy as np

from cps.pythia.planner import damping_family, plan_scalar_control


def test_planner_returns_candidate():
    matrix = np.array([[1.05, 0.2], [0.0, 0.8]], dtype=np.complex128)
    recommendation = plan_scalar_control(
        "damping",
        0.0,
        [0.0, 0.1, 0.2],
        lambda value: damping_family(matrix, value),
        [(0, 1)],
    )
    assert recommendation.recommended in {0.0, 0.1, 0.2}
    assert recommendation.recommended_risk <= recommendation.baseline_risk


def test_damping_family_is_contracting_surrogate():
    matrix = np.array([[1.1, 0.3], [0.2, 0.9]], dtype=np.complex128)
    damped = damping_family(matrix, 0.25)
    assert np.allclose(damped, 0.75 * matrix)
    assert np.linalg.norm(damped, 2) < np.linalg.norm(matrix, 2)
