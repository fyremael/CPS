import numpy as np
import pytest

from cps.subspace_stability import (
    leading_eigenspace_stability,
    leading_singular_subspace_stability,
    summarize_singular_subspace_sweep,
)


def test_tran_vu_eigenspace_certificate_can_improve_classical_bound():
    reference = np.diag([1000.0, 996.0, 0.0, -1.0])
    perturbation = np.zeros_like(reference)
    perturbation[0, 2] = perturbation[2, 0] = 0.5
    certificate = leading_eigenspace_stability(reference, reference + perturbation)

    assert certificate.halving_rank == 2
    assert certificate.directional_coupling == pytest.approx(0.0)
    assert certificate.theorem_applicable
    assert certificate.informative
    assert certificate.improves_classical
    assert certificate.admitted
    assert certificate.tran_vu_bound < certificate.davis_kahan_bound
    assert certificate.observed_projector_distance <= certificate.tran_vu_bound


def test_certificate_fails_closed_when_gap_is_below_moderate_gap_threshold():
    reference = np.diag([1000.0, 996.0, 0.0])
    perturbation = np.zeros_like(reference)
    perturbation[0, 2] = perturbation[2, 0] = 2.0
    certificate = leading_eigenspace_stability(reference, reference + perturbation)

    assert not certificate.moderate_gap_lower_holds
    assert not certificate.theorem_applicable
    assert not certificate.admitted
    assert "gap_below_four_perturbation_norms" in certificate.reasons


def test_singular_space_certificate_reports_left_and_right_distances():
    reference = np.diag([1000.0, 996.0, 0.0])
    perturbation = np.zeros_like(reference)
    perturbation[0, 2] = 0.5
    certificate = leading_singular_subspace_stability(reference, reference + perturbation)

    assert certificate.target == "leading_singular_space"
    assert certificate.observed_left_projector_distance is not None
    assert certificate.observed_right_projector_distance is not None
    assert certificate.observed_projector_distance == max(
        certificate.observed_left_projector_distance,
        certificate.observed_right_projector_distance,
    )
    assert certificate.theorem_applicable


def test_complex_singular_problem_uses_realification():
    reference = np.diag([1000.0, 996.0, 0.0]).astype(complex)
    perturbed = reference.copy()
    perturbed[0, 2] = 0.5j
    certificate = leading_singular_subspace_stability(reference, perturbed)

    assert certificate.representation == "complex_realification"
    assert certificate.working_rank == 2
    assert certificate.gap == pytest.approx(4.0)
    assert certificate.perturbation_norm == pytest.approx(0.5)


def test_complex_input_can_be_rejected_explicitly():
    reference = np.diag([3.0, 2.0, 0.0]).astype(complex)
    perturbed = reference.copy()
    perturbed[0, 2] = 0.1j
    with pytest.raises(ValueError, match="real matrices"):
        leading_singular_subspace_stability(reference, perturbed, complex_policy="reject")


def test_sweep_summary_skips_duplicate_endpoint_and_counts_admissions():
    reference = np.diag([1000.0, 996.0, 0.0]).astype(complex)
    first = reference.copy()
    first[0, 2] = 0.5j
    second = reference.copy()
    second[0, 2] = -0.5j
    summary = summarize_singular_subspace_sweep(
        [reference, first, second, reference.copy()]
    )

    assert summary["comparison_count"] == 2
    assert summary["skipped_near_reference_count"] == 1
    assert summary["theorem_applicable_count"] == 2
    assert summary["admitted_count"] == 2
    assert summary["most_informative_comparison"] is not None


def test_invalid_rank_is_rejected():
    matrix = np.eye(2)
    with pytest.raises(ValueError, match="rank"):
        leading_singular_subspace_stability(matrix, matrix, rank=2)
