from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class CPSMetrics:
    spectral_radius_max: float
    spectral_radius_min: float
    spectral_abscissa_max: float
    minimum_gap: float
    maximum_eigenvalue_condition: float
    finite_horizon_gain: float
    kreiss_surrogate: float
    total_loop_length: float
    total_signed_loop_area: float
    baseline_displacement_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def minimum_eigenvalue_gap(values: ArrayLike) -> float:
    eig = np.asarray(values, dtype=np.complex128).reshape(-1)
    if eig.size < 2:
        return float("inf")
    distances = np.abs(eig[:, None] - eig[None, :])
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def eigenvalue_condition_numbers(matrix: ArrayLike) -> NDArray[np.float64]:
    """Condition numbers of simple eigenvalues from left/right eigenvectors."""

    a = np.asarray(matrix, dtype=np.complex128)
    values_r, right = np.linalg.eig(a)
    values_l, left_raw = np.linalg.eig(a.conj().T)
    result = np.empty(values_r.size, dtype=np.float64)
    for k, value in enumerate(values_r):
        j = int(np.argmin(np.abs(values_l.conj() - value)))
        x = right[:, k]
        y = left_raw[:, j]
        denom = abs(np.vdot(y, x))
        result[k] = (
            float(np.linalg.norm(x) * np.linalg.norm(y) / denom)
            if denom > np.finfo(float).eps
            else float("inf")
        )
    return result


def finite_horizon_gain(matrix: ArrayLike, horizon: int = 20) -> float:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    a = np.asarray(matrix, dtype=np.complex128)
    power = np.eye(a.shape[0], dtype=np.complex128)
    best = 1.0
    for _ in range(horizon):
        power = power @ a
        best = max(best, float(np.linalg.svd(power, compute_uv=False)[0]))
    return best


def kreiss_surrogate(
    matrix: ArrayLike,
    radii: ArrayLike | None = None,
    angles: int = 64,
) -> float:
    """Coarse discrete-time Kreiss constant surrogate.

    Computes max_{|z|>1} (|z|-1) ||(zI-A)^{-1}|| over a polar grid.
    This is a diagnostic, not a certified bound.
    """

    a = np.asarray(matrix, dtype=np.complex128)
    if radii is None:
        radii = np.geomspace(1.001, 2.5, 24)
    radii = np.asarray(radii, dtype=float)
    if np.any(radii <= 1.0):
        raise ValueError("all radii must exceed one")
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    identity = np.eye(a.shape[0], dtype=np.complex128)
    best = 0.0
    for radius in radii:
        for angle in theta:
            z = radius * np.exp(1j * angle)
            try:
                smallest = np.linalg.svd(z * identity - a, compute_uv=False)[-1]
            except np.linalg.LinAlgError:
                return float("inf")
            if smallest <= np.finfo(float).eps:
                return float("inf")
            best = max(best, float((radius - 1.0) / smallest))
    return best


def _loop_geometry(tracked_eigenvalues: ComplexMatrix) -> tuple[float, float]:
    total_length = 0.0
    total_area = 0.0
    for branch in tracked_eigenvalues.T:
        closed = np.concatenate([branch, branch[:1]])
        total_length += float(np.sum(np.abs(np.diff(closed))))
        x, y = closed.real, closed.imag
        total_area += float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return total_length, total_area


def compute_cps_metrics(
    matrices: list[ComplexMatrix],
    tracked_eigenvalues: ComplexMatrix,
    finite_horizon: int = 20,
    compute_kreiss: bool = True,
) -> CPSMetrics:
    if not matrices:
        raise ValueError("matrices must not be empty")
    radii = np.max(np.abs(tracked_eigenvalues), axis=1)
    abscissae = np.max(tracked_eigenvalues.real, axis=1)
    min_gap = min(minimum_eigenvalue_gap(row) for row in tracked_eigenvalues)

    max_condition = 0.0
    max_gain = 1.0
    max_kreiss = float("nan") if not compute_kreiss else 0.0
    for matrix in matrices:
        conditions = eigenvalue_condition_numbers(matrix)
        finite_conditions = conditions[np.isfinite(conditions)]
        if np.any(~np.isfinite(conditions)):
            max_condition = float("inf")
        elif finite_conditions.size and np.isfinite(max_condition):
            max_condition = max(max_condition, float(np.max(finite_conditions)))
        max_gain = max(max_gain, finite_horizon_gain(matrix, finite_horizon))
        if compute_kreiss:
            max_kreiss = max(max_kreiss, kreiss_surrogate(matrix))

    length, area = _loop_geometry(tracked_eigenvalues)
    baseline = tracked_eigenvalues[0]
    baseline_displacement = float(
        np.max(np.abs(tracked_eigenvalues - baseline[None, :]))
    )
    return CPSMetrics(
        spectral_radius_max=float(np.max(radii)),
        spectral_radius_min=float(np.min(radii)),
        spectral_abscissa_max=float(np.max(abscissae)),
        minimum_gap=float(min_gap),
        maximum_eigenvalue_condition=float(max_condition),
        finite_horizon_gain=float(max_gain),
        kreiss_surrogate=float(max_kreiss),
        total_loop_length=float(length),
        total_signed_loop_area=float(area),
        baseline_displacement_max=baseline_displacement,
    )
