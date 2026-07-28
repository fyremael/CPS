from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class RegimeFixture:
    """A deterministic matrix pair with declared certificate expectations."""

    name: str
    title: str
    description: str
    reference: ComplexMatrix
    perturbed: ComplexMatrix
    rank: int
    expected_theorem_applicable: bool
    expected_informative: bool
    expected_improves_classical: bool
    expected_admitted: bool
    expected_reason: str | None = None
    expected_representation: str | None = None
    minimum_halving_rank: int | None = None
    tags: tuple[str, ...] = ()


def rotation(dimension: int, first: int, second: int, angle: float) -> ComplexMatrix:
    matrix = np.eye(dimension, dtype=np.complex128)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix[first, first] = cosine
    matrix[second, second] = cosine
    matrix[first, second] = -sine
    matrix[second, first] = sine
    return matrix


def _diagonal(values: list[float]) -> ComplexMatrix:
    return np.diag(np.asarray(values, dtype=np.complex128))


def _perturbed(reference: ComplexMatrix, entries: dict[tuple[int, int], complex]) -> ComplexMatrix:
    output = reference.copy()
    for (row, column), value in entries.items():
        output[row, column] += value
    return output


def regime_fixtures() -> tuple[RegimeFixture, ...]:
    """Return the governed regime matrix used by the characterization package."""

    moderate = _diagonal([1000.0, 996.0, 0.0])
    weak = _perturbed(moderate, {(0, 2): 0.5})

    large_gap = _diagonal([1000.0, 900.0, 0.0])
    large_gap_perturbed = _perturbed(large_gap, {(0, 2): 0.1})

    local = 0.03
    remote = sqrt(0.5**2 - local**2)
    vacuous = _perturbed(moderate, {(0, 1): local, (0, 2): remote})

    gap_failure = _perturbed(moderate, {(0, 2): 1.1})

    small_signal = _diagonal([4.0, 2.5, 0.0])
    small_signal_perturbed = _perturbed(small_signal, {(0, 2): 0.1})

    strong_local = _perturbed(moderate, {(0, 1): 0.5})

    clustered = _diagonal([1000.0, 996.0, 995.0, 994.0, 993.0, 0.0])
    cluster_local = 0.01
    cluster_remote = sqrt(0.5**2 - cluster_local**2)
    clustered_perturbed = _perturbed(
        clustered,
        {(0, 1): cluster_local, (0, 5): cluster_remote},
    )

    complex_reference = moderate.astype(np.complex128)
    complex_perturbed = _perturbed(complex_reference, {(0, 2): 0.5j})

    left = rotation(3, 0, 1, 0.4)
    right = rotation(3, 1, 2, -0.6)
    nonnormal = left @ _diagonal([1000.0, 996.0, 0.0]) @ right.conj().T
    nonnormal_perturbed = nonnormal + 0.5 * np.outer(left[:, 0], right[:, 2].conj())

    rank_two_right = rotation(4, 0, 3, 0.3)
    rank_two = _diagonal([1000.0, 999.0, 995.0, 0.0]) @ rank_two_right.conj().T
    rank_two_perturbed = rank_two + 0.5 * np.outer(
        np.eye(4, dtype=np.complex128)[:, 0],
        rank_two_right[:, 3].conj(),
    )

    return (
        RegimeFixture(
            name="weak_directional_coupling",
            title="Weak directional coupling",
            description=(
                "The perturbation is globally visible but misses the nearby signal block, "
                "so Tran--Vu should be valid, informative, and sharper than Davis--Kahan."
            ),
            reference=moderate,
            perturbed=weak,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=True,
            expected_improves_classical=True,
            expected_admitted=True,
            tags=("intended-improvement", "weak-local-coupling"),
        ),
        RegimeFixture(
            name="classical_sharper_large_gap",
            title="Classical large-gap regime",
            description=(
                "The target gap is large enough that classical Davis--Kahan is already tighter "
                "than the moderate-gap refinement."
            ),
            reference=large_gap,
            perturbed=large_gap_perturbed,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=True,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="tran_vu_not_sharper_than_davis_kahan",
            tags=("classical-preferred", "large-gap"),
        ),
        RegimeFixture(
            name="valid_but_vacuous",
            title="Valid but vacuous",
            description=(
                "The theorem applies, but a small amount of local coupling makes the stated "
                "upper bound exceed one."
            ),
            reference=moderate,
            perturbed=vacuous,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=False,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="tran_vu_bound_not_informative",
            tags=("null-result", "vacuous-bound"),
        ),
        RegimeFixture(
            name="gap_failure",
            title="Moderate-gap failure",
            description="The perturbation exceeds one quarter of the target gap.",
            reference=moderate,
            perturbed=gap_failure,
            rank=1,
            expected_theorem_applicable=False,
            expected_informative=False,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="gap_below_four_perturbation_norms",
            tags=("fail-closed", "gap-condition"),
        ),
        RegimeFixture(
            name="signal_scale_failure",
            title="Signal-scale failure",
            description=(
                "The local gap is too large relative to the target signal singular value for "
                "the moderate-gap theorem."
            ),
            reference=small_signal,
            perturbed=small_signal_perturbed,
            rank=1,
            expected_theorem_applicable=False,
            expected_informative=False,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="gap_above_signal_quarter",
            tags=("fail-closed", "signal-scale-condition"),
        ),
        RegimeFixture(
            name="strong_local_coupling",
            title="Strong local coupling",
            description=(
                "The perturbation acts directly inside the nearby signal block, eliminating the "
                "directional advantage."
            ),
            reference=moderate,
            perturbed=strong_local,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=False,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="tran_vu_bound_not_informative",
            tags=("strong-local-coupling", "null-result"),
        ),
        RegimeFixture(
            name="high_halving_rank_penalty",
            title="High halving-rank penalty",
            description=(
                "A broad near-signal cluster increases the r-squared penalty even though the "
                "local directional coupling is numerically small."
            ),
            reference=clustered,
            perturbed=clustered_perturbed,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=False,
            expected_improves_classical=False,
            expected_admitted=False,
            expected_reason="tran_vu_bound_not_informative",
            minimum_halving_rank=5,
            tags=("cluster-width", "halving-rank"),
        ),
        RegimeFixture(
            name="complex_realification",
            title="Complex phase realification",
            description=(
                "A genuinely imaginary coupling is mapped to the real block representation while "
                "preserving the operator norm and duplicated singular spectrum."
            ),
            reference=complex_reference,
            perturbed=complex_perturbed,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=True,
            expected_improves_classical=True,
            expected_admitted=True,
            expected_representation="complex_realification",
            tags=("complex-phase", "realification"),
        ),
        RegimeFixture(
            name="nonnormal_singular_stability",
            title="Non-normal singular stability",
            description=(
                "The operator is deliberately non-normal, but its leading singular channel remains "
                "inside the admitted Tran--Vu regime."
            ),
            reference=nonnormal,
            perturbed=nonnormal_perturbed,
            rank=1,
            expected_theorem_applicable=True,
            expected_informative=True,
            expected_improves_classical=True,
            expected_admitted=True,
            tags=("non-normal", "singular-space"),
        ),
        RegimeFixture(
            name="rank_two_cluster",
            title="Rank-two signal cluster",
            description=(
                "The target object is a two-dimensional leading singular subspace rather than a "
                "single vector."
            ),
            reference=rank_two,
            perturbed=rank_two_perturbed,
            rank=2,
            expected_theorem_applicable=True,
            expected_informative=True,
            expected_improves_classical=True,
            expected_admitted=True,
            tags=("rank-two", "cluster-subspace"),
        ),
    )
