from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import log, pi, sqrt
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

RealMatrix = NDArray[np.float64]
ComplexMatrix = NDArray[np.complex128]
ComplexPolicy = Literal["realify", "reject"]


@dataclass(frozen=True)
class SubspaceStabilityCertificate:
    """A classical and Tran--Vu comparison for one subspace perturbation.

    ``admitted`` is a CPS policy field. It is true only when the theorem's
    hypotheses hold and the resulting Tran--Vu bound is both informative
    (strictly below one) and sharper than the classical Davis--Kahan bound.
    """

    target: str
    rank: int
    working_rank: int
    representation: str
    reference_shape: tuple[int, int]
    perturbation_norm: float
    gap: float
    signal: float
    spectral_scale: float
    halving_rank: int | None
    directional_coupling: float | None
    directional_coupling_ratio: float | None
    davis_kahan_bound: float | None
    tran_vu_bound: float | None
    observed_projector_distance: float
    observed_left_projector_distance: float | None
    observed_right_projector_distance: float | None
    moderate_gap_lower_holds: bool
    moderate_gap_upper_holds: bool
    theorem_applicable: bool
    improves_classical: bool
    informative: bool
    admitted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_matrix(value: ArrayLike, *, name: str) -> ComplexMatrix:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite entries")
    return matrix


def _validate_pair(reference: ArrayLike, perturbed: ArrayLike) -> tuple[ComplexMatrix, ComplexMatrix]:
    base = _as_matrix(reference, name="reference")
    noisy = _as_matrix(perturbed, name="perturbed")
    if base.shape != noisy.shape:
        raise ValueError("reference and perturbed must have the same shape")
    return base, noisy


def _realification(matrix: ComplexMatrix) -> RealMatrix:
    """Real block representation preserving operator and singular-value norms."""

    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


def _working_pair(
    reference: ComplexMatrix,
    perturbed: ComplexMatrix,
    rank: int,
    *,
    complex_policy: ComplexPolicy,
    imaginary_tolerance: float,
) -> tuple[RealMatrix, RealMatrix, int, str]:
    scale = max(
        1.0,
        float(np.linalg.norm(reference, ord=2)),
        float(np.linalg.norm(perturbed, ord=2)),
    )
    imaginary_size = max(
        float(np.max(np.abs(reference.imag))),
        float(np.max(np.abs(perturbed.imag))),
    )
    if imaginary_size <= imaginary_tolerance * scale:
        return reference.real, perturbed.real, rank, "real"
    if complex_policy == "reject":
        raise ValueError(
            "Tran--Vu certificates require real matrices unless complex_policy='realify'"
        )
    if complex_policy != "realify":
        raise ValueError(f"unknown complex_policy: {complex_policy}")
    return _realification(reference), _realification(perturbed), 2 * rank, "complex_realification"


def _halving_rank(values: NDArray[np.float64], rank: int) -> int | None:
    """Return the theorem's one-based halving rank r, or None if unavailable."""

    signal = abs(float(values[rank - 1]))
    for r in range(rank, values.size):
        next_value = float(values[r])  # values[r] is lambda_{r+1} in one-based notation.
        if signal / 2.0 <= abs(float(values[rank - 1]) - next_value):
            return r
    return None


def _projector_distance(first: ComplexMatrix, second: ComplexMatrix) -> float:
    projector_first = first @ first.conj().T
    projector_second = second @ second.conj().T
    return float(np.linalg.norm(projector_second - projector_first, ord=2))


def _safe_bound(numerator: float, denominator: float, constant: float = 1.0) -> float | None:
    if denominator <= 0.0:
        return None
    return float(constant * numerator / denominator)


def _build_certificate(
    *,
    target: str,
    rank: int,
    working_rank: int,
    representation: str,
    reference_shape: tuple[int, int],
    perturbation_norm: float,
    gap: float,
    signal: float,
    spectral_scale: float,
    halving_rank: int | None,
    directional_coupling: float | None,
    davis_kahan_bound: float | None,
    tran_vu_constant: float,
    observed_projector_distance: float,
    observed_left_projector_distance: float | None = None,
    observed_right_projector_distance: float | None = None,
) -> SubspaceStabilityCertificate:
    lower_holds = gap > 0.0 and 4.0 * perturbation_norm <= gap
    upper_holds = signal > 0.0 and gap <= signal / 4.0

    tran_vu_bound: float | None = None
    if (
        gap > 0.0
        and signal > 0.0
        and spectral_scale > 0.0
        and halving_rank is not None
        and directional_coupling is not None
    ):
        log_argument = 6.0 * spectral_scale / gap
        if log_argument > 1.0:
            tran_vu_bound = float(
                tran_vu_constant
                * (
                    perturbation_norm / signal * log(log_argument)
                    + halving_rank**2 * directional_coupling / gap
                )
            )

    theorem_applicable = lower_holds and upper_holds and tran_vu_bound is not None
    improves_classical = (
        theorem_applicable
        and davis_kahan_bound is not None
        and tran_vu_bound is not None
        and tran_vu_bound < davis_kahan_bound
    )
    informative = theorem_applicable and tran_vu_bound is not None and tran_vu_bound < 1.0
    admitted = theorem_applicable and improves_classical and informative

    reasons: list[str] = []
    if gap <= 0.0:
        reasons.append("nonpositive_target_gap")
    if signal <= 0.0:
        reasons.append("nonpositive_signal")
    if halving_rank is None:
        reasons.append("halving_rank_unavailable")
    if gap > 0.0 and not lower_holds:
        reasons.append("gap_below_four_perturbation_norms")
    if signal > 0.0 and not upper_holds:
        reasons.append("gap_above_signal_quarter")
    if theorem_applicable and not informative:
        reasons.append("tran_vu_bound_not_informative")
    if theorem_applicable and not improves_classical:
        reasons.append("tran_vu_not_sharper_than_davis_kahan")

    coupling_ratio = None
    if directional_coupling is not None:
        coupling_ratio = (
            0.0
            if perturbation_norm == 0.0 and directional_coupling == 0.0
            else _safe_bound(directional_coupling, perturbation_norm)
        )

    return SubspaceStabilityCertificate(
        target=target,
        rank=rank,
        working_rank=working_rank,
        representation=representation,
        reference_shape=reference_shape,
        perturbation_norm=perturbation_norm,
        gap=gap,
        signal=signal,
        spectral_scale=spectral_scale,
        halving_rank=halving_rank,
        directional_coupling=directional_coupling,
        directional_coupling_ratio=coupling_ratio,
        davis_kahan_bound=davis_kahan_bound,
        tran_vu_bound=tran_vu_bound,
        observed_projector_distance=observed_projector_distance,
        observed_left_projector_distance=observed_left_projector_distance,
        observed_right_projector_distance=observed_right_projector_distance,
        moderate_gap_lower_holds=lower_holds,
        moderate_gap_upper_holds=upper_holds,
        theorem_applicable=theorem_applicable,
        improves_classical=improves_classical,
        informative=informative,
        admitted=admitted,
        reasons=tuple(reasons),
    )


def leading_eigenspace_stability(
    reference: ArrayLike,
    perturbed: ArrayLike,
    *,
    rank: int = 1,
    complex_policy: ComplexPolicy = "realify",
    imaginary_tolerance: float = 1e-12,
    symmetry_tolerance: float = 1e-10,
) -> SubspaceStabilityCertificate:
    """Certify a leading Hermitian eigenspace using Tran--Vu Theorem 2.1.

    Complex Hermitian matrices are represented by their real block form. The
    reported ``working_rank`` is therefore twice ``rank`` in that case.
    """

    base, noisy = _validate_pair(reference, perturbed)
    if base.shape[0] != base.shape[1]:
        raise ValueError("leading eigenspace certificates require square matrices")
    if not (1 <= rank < base.shape[0]):
        raise ValueError("rank must satisfy 1 <= rank < matrix dimension")
    scale = max(
        1.0,
        float(np.linalg.norm(base, ord=2)),
        float(np.linalg.norm(noisy, ord=2)),
    )
    if float(np.linalg.norm(base - base.conj().T, ord=2)) > symmetry_tolerance * scale:
        raise ValueError("reference must be Hermitian")
    if float(np.linalg.norm(noisy - noisy.conj().T, ord=2)) > symmetry_tolerance * scale:
        raise ValueError("perturbed must be Hermitian")

    work_base, work_noisy, work_rank, representation = _working_pair(
        base,
        noisy,
        rank,
        complex_policy=complex_policy,
        imaginary_tolerance=imaginary_tolerance,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(work_base)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    noisy_values, noisy_vectors = np.linalg.eigh(work_noisy)
    noisy_order = np.argsort(noisy_values)[::-1]
    noisy_vectors = noisy_vectors[:, noisy_order]

    perturbation = work_noisy - work_base
    perturbation_norm = float(np.linalg.norm(perturbation, ord=2))
    gap = float(eigenvalues[work_rank - 1] - eigenvalues[work_rank])
    signal = abs(float(eigenvalues[work_rank - 1]))
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    halving_rank = _halving_rank(eigenvalues, work_rank)
    directional_coupling = None
    if halving_rank is not None:
        local = eigenvectors[:, :halving_rank]
        directional_coupling = float(np.max(np.abs(local.T @ perturbation @ local)))

    observed_working = _projector_distance(
        eigenvectors[:, :work_rank].astype(np.complex128),
        noisy_vectors[:, :work_rank].astype(np.complex128),
    )
    base_values_original, base_vectors_original = np.linalg.eigh(base)
    noisy_values_original, noisy_vectors_original = np.linalg.eigh(noisy)
    base_order_original = np.argsort(base_values_original)[::-1]
    noisy_order_original = np.argsort(noisy_values_original)[::-1]
    observed_original = _projector_distance(
        base_vectors_original[:, base_order_original[:rank]],
        noisy_vectors_original[:, noisy_order_original[:rank]],
    )
    observed = observed_original if representation == "complex_realification" else observed_working

    return _build_certificate(
        target="leading_eigenspace",
        rank=rank,
        working_rank=work_rank,
        representation=representation,
        reference_shape=base.shape,
        perturbation_norm=perturbation_norm,
        gap=gap,
        signal=signal,
        spectral_scale=spectral_scale,
        halving_rank=halving_rank,
        directional_coupling=directional_coupling,
        davis_kahan_bound=_safe_bound(perturbation_norm, gap, pi),
        tran_vu_constant=24.0,
        observed_projector_distance=observed,
    )


def leading_singular_subspace_stability(
    reference: ArrayLike,
    perturbed: ArrayLike,
    *,
    rank: int = 1,
    complex_policy: ComplexPolicy = "realify",
    imaginary_tolerance: float = 1e-12,
) -> SubspaceStabilityCertificate:
    """Certify leading left/right singular spaces using Tran--Vu Theorem 2.3."""

    base, noisy = _validate_pair(reference, perturbed)
    maximum_rank = min(base.shape)
    if not (1 <= rank < maximum_rank):
        raise ValueError("rank must satisfy 1 <= rank < min(matrix shape)")

    work_base, work_noisy, work_rank, representation = _working_pair(
        base,
        noisy,
        rank,
        complex_policy=complex_policy,
        imaginary_tolerance=imaginary_tolerance,
    )
    left, singular_values, right_h = np.linalg.svd(work_base, full_matrices=False)
    noisy_left, _, noisy_right_h = np.linalg.svd(work_noisy, full_matrices=False)
    perturbation = work_noisy - work_base
    perturbation_norm = float(np.linalg.norm(perturbation, ord=2))
    gap = float(singular_values[work_rank - 1] - singular_values[work_rank])
    signal = float(singular_values[work_rank - 1])
    spectral_scale = float(singular_values[0])
    halving_rank = _halving_rank(singular_values, work_rank)
    directional_coupling = None
    if halving_rank is not None:
        local_left = left[:, :halving_rank]
        local_right = right_h.conj().T[:, :halving_rank]
        directional_coupling = float(
            np.max(np.abs(local_left.T @ perturbation @ local_right))
        )

    working_left_distance = _projector_distance(
        left[:, :work_rank].astype(np.complex128),
        noisy_left[:, :work_rank].astype(np.complex128),
    )
    working_right_distance = _projector_distance(
        right_h.conj().T[:, :work_rank].astype(np.complex128),
        noisy_right_h.conj().T[:, :work_rank].astype(np.complex128),
    )

    original_left, _, original_right_h = np.linalg.svd(base, full_matrices=False)
    original_noisy_left, _, original_noisy_right_h = np.linalg.svd(noisy, full_matrices=False)
    left_distance = _projector_distance(original_left[:, :rank], original_noisy_left[:, :rank])
    right_distance = _projector_distance(
        original_right_h.conj().T[:, :rank],
        original_noisy_right_h.conj().T[:, :rank],
    )
    if representation == "real":
        left_distance = working_left_distance
        right_distance = working_right_distance

    return _build_certificate(
        target="leading_singular_space",
        rank=rank,
        working_rank=work_rank,
        representation=representation,
        reference_shape=base.shape,
        perturbation_norm=perturbation_norm,
        gap=gap,
        signal=signal,
        spectral_scale=spectral_scale,
        halving_rank=halving_rank,
        directional_coupling=directional_coupling,
        davis_kahan_bound=_safe_bound(perturbation_norm, gap, pi),
        tran_vu_constant=24.0 * sqrt(2.0),
        observed_projector_distance=max(left_distance, right_distance),
        observed_left_projector_distance=left_distance,
        observed_right_projector_distance=right_distance,
    )


def summarize_singular_subspace_sweep(
    matrices: Sequence[ArrayLike],
    *,
    rank: int = 1,
    reference_index: int = 0,
    complex_policy: ComplexPolicy = "realify",
    near_reference_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Summarize Tran--Vu certificates over a matrix sweep.

    The reference sample and numerically duplicate endpoint samples are excluded
    from admission counts. The returned summary is JSON-serializable.
    """

    if len(matrices) < 2:
        raise ValueError("matrices must contain at least two samples")
    if not (0 <= reference_index < len(matrices)):
        raise IndexError("reference_index is out of range")
    reference = _as_matrix(matrices[reference_index], name="reference")
    reference_norm = float(np.linalg.norm(reference, ord=2))
    threshold = near_reference_tolerance * max(1.0, reference_norm)

    entries: list[tuple[int, SubspaceStabilityCertificate]] = []
    skipped_near_reference = 0
    for index, matrix in enumerate(matrices):
        if index == reference_index:
            continue
        candidate = _as_matrix(matrix, name=f"matrices[{index}]")
        if candidate.shape != reference.shape:
            raise ValueError("all sweep matrices must share one shape")
        if float(np.linalg.norm(candidate - reference, ord=2)) <= threshold:
            skipped_near_reference += 1
            continue
        entries.append(
            (
                index,
                leading_singular_subspace_stability(
                    reference,
                    candidate,
                    rank=rank,
                    complex_policy=complex_policy,
                ),
            )
        )

    reason_counts: Counter[str] = Counter()
    for _, certificate in entries:
        reason_counts.update(certificate.reasons)

    applicable = [(index, cert) for index, cert in entries if cert.theorem_applicable]
    admitted = [(index, cert) for index, cert in entries if cert.admitted]
    improving = [(index, cert) for index, cert in entries if cert.improves_classical]

    informative_entry = None
    candidates = [
        (index, cert)
        for index, cert in applicable
        if cert.tran_vu_bound is not None
    ]
    if candidates:
        informative_index, informative_certificate = min(
            candidates,
            key=lambda item: float(item[1].tran_vu_bound),
        )
        informative_entry = {
            "index": informative_index,
            "certificate": informative_certificate.to_dict(),
        }

    def maximum(
        field: str,
        rows: Sequence[tuple[int, SubspaceStabilityCertificate]],
    ) -> float | None:
        values = [getattr(certificate, field) for _, certificate in rows]
        finite = [
            float(value)
            for value in values
            if value is not None and np.isfinite(value)
        ]
        return max(finite) if finite else None

    best_ratio = None
    ratios = []
    for _, certificate in applicable:
        if (
            certificate.tran_vu_bound is not None
            and certificate.davis_kahan_bound is not None
            and certificate.davis_kahan_bound > 0.0
        ):
            ratios.append(certificate.tran_vu_bound / certificate.davis_kahan_bound)
    if ratios:
        best_ratio = float(min(ratios))

    representation = entries[0][1].representation if entries else "unavailable"
    working_rank = entries[0][1].working_rank if entries else None
    return {
        "theorem": "Tran--Vu (2025), Theorem 2.3",
        "target": "leading_singular_space",
        "rank": rank,
        "working_rank": working_rank,
        "representation": representation,
        "reference_index": reference_index,
        "sample_count": len(matrices),
        "comparison_count": len(entries),
        "skipped_near_reference_count": skipped_near_reference,
        "theorem_applicable_count": len(applicable),
        "improvement_count": len(improving),
        "admitted_count": len(admitted),
        "maximum_observed_projector_distance": maximum(
            "observed_projector_distance", entries
        ),
        "maximum_davis_kahan_bound": maximum("davis_kahan_bound", entries),
        "maximum_tran_vu_bound_when_applicable": maximum(
            "tran_vu_bound", applicable
        ),
        "best_tran_vu_to_davis_kahan_ratio": best_ratio,
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "most_informative_comparison": informative_entry,
    }
