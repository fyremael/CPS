from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import CPSMetrics, compute_cps_metrics
from .tracking import track_eigenpairs

ComplexMatrix = NDArray[np.complex128]
MatrixFamily = Callable[[float], ComplexMatrix]


@dataclass(frozen=True)
class FamilySweepResult:
    """Spectrum and metrics for an arbitrary one-parameter matrix family."""

    phases: NDArray[np.float64]
    matrices: list[ComplexMatrix]
    eigenvalues: ComplexMatrix
    eigenvectors: list[ComplexMatrix]
    metrics: CPSMetrics
    family_name: str
    metadata: dict[str, object]


def sweep_matrix_family(
    family: MatrixFamily,
    phases: Iterable[float],
    *,
    family_name: str,
    finite_horizon: int = 20,
    compute_kreiss: bool = True,
    metadata: dict[str, object] | None = None,
) -> FamilySweepResult:
    phase_array = np.asarray(list(phases), dtype=float)
    if phase_array.ndim != 1 or phase_array.size < 2:
        raise ValueError("phases must be one-dimensional with at least two samples")

    matrices: list[ComplexMatrix] = []
    raw_values: list[NDArray[np.complex128]] = []
    raw_vectors: list[ComplexMatrix] = []
    shape: tuple[int, int] | None = None
    for phase in phase_array:
        matrix = np.asarray(family(float(phase)), dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("family must return square matrices")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("family returned inconsistent matrix shapes")
        if not np.isfinite(matrix).all():
            raise ValueError("family returned non-finite entries")
        values, vectors = np.linalg.eig(matrix)
        matrices.append(matrix)
        raw_values.append(values)
        raw_vectors.append(vectors)

    tracked_values, tracked_vectors = track_eigenpairs(raw_values, raw_vectors)
    metrics = compute_cps_metrics(
        matrices,
        tracked_values,
        finite_horizon=finite_horizon,
        compute_kreiss=compute_kreiss,
    )
    return FamilySweepResult(
        phases=phase_array,
        matrices=matrices,
        eigenvalues=tracked_values,
        eigenvectors=tracked_vectors,
        metrics=metrics,
        family_name=family_name,
        metadata={} if metadata is None else dict(metadata),
    )


def singular_channel_family(
    matrix: ArrayLike,
    rows: slice | list[int] | NDArray[np.integer],
    cols: slice | list[int] | NDArray[np.integer],
    channel: int = 0,
) -> tuple[MatrixFamily, dict[str, object]]:
    """Return a phase family for one singular channel of a coupling block.

    If ``B = U diag(s) V*`` is the selected block, the family replaces only
    ``s[channel] u[channel] v[channel]*`` by its phase-rotated counterpart.
    Singular values and all unselected block channels remain fixed.
    """

    base = np.asarray(matrix, dtype=np.complex128)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("matrix must be square")
    row_index = rows
    col_index = cols
    if not isinstance(rows, slice) and not isinstance(cols, slice):
        row_index, col_index = np.ix_(np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))
    block = base[row_index, col_index]
    if block.ndim != 2:
        raise ValueError("selected block must be two-dimensional")
    u, singular_values, vh = np.linalg.svd(block, full_matrices=False)
    if not (0 <= channel < singular_values.size):
        raise IndexError("singular channel out of range")
    sigma = float(singular_values[channel])
    rank_one = sigma * np.outer(u[:, channel], vh[channel, :])

    def family(phase: float) -> ComplexMatrix:
        out = base.copy()
        out[row_index, col_index] = block + (np.exp(1j * phase) - 1.0) * rank_one
        return out

    metadata: dict[str, object] = {
        "channel": int(channel),
        "singular_value": sigma,
        "block_shape": list(block.shape),
    }
    return family, metadata
