from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import CPSMetrics, compute_cps_metrics
from .perturbations import phase_sweep_entry
from .tracking import track_eigenpairs


ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class SweepResult:
    phases: NDArray[np.float64]
    matrices: list[ComplexMatrix]
    eigenvalues: ComplexMatrix
    eigenvectors: list[ComplexMatrix]
    metrics: CPSMetrics
    row: int
    col: int


def spectral_sweep(
    matrix: ArrayLike,
    row: int,
    col: int,
    phases: Iterable[float] | None = None,
    finite_horizon: int = 20,
    compute_kreiss: bool = True,
) -> SweepResult:
    if phases is None:
        phases = np.linspace(0.0, 2.0 * np.pi, 65)
    phase_array = np.asarray(list(phases), dtype=float)
    if phase_array.ndim != 1 or phase_array.size < 2:
        raise ValueError("phases must be a one-dimensional sequence with at least two values")

    matrices = phase_sweep_entry(matrix, row, col, phase_array)
    raw_values: list[NDArray[np.complex128]] = []
    raw_vectors: list[ComplexMatrix] = []
    for current in matrices:
        values, vectors = np.linalg.eig(current)
        raw_values.append(values)
        raw_vectors.append(vectors)
    tracked_values, tracked_vectors = track_eigenpairs(raw_values, raw_vectors)
    metrics = compute_cps_metrics(
        matrices,
        tracked_values,
        finite_horizon=finite_horizon,
        compute_kreiss=compute_kreiss,
    )
    return SweepResult(
        phases=phase_array,
        matrices=matrices,
        eigenvalues=tracked_values,
        eigenvectors=tracked_vectors,
        metrics=metrics,
        row=row,
        col=col,
    )
