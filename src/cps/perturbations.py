from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class EntryPhaseFamily:
    """A magnitude-preserving phase family for one matrix entry."""

    matrix: ComplexMatrix
    row: int
    col: int
    baseline_phase: float
    magnitude: float

    def at(self, phase_offset: float) -> ComplexMatrix:
        out = self.matrix.copy()
        out[self.row, self.col] = self.magnitude * np.exp(
            1j * (self.baseline_phase + phase_offset)
        )
        return out


def _as_square_complex(matrix: ArrayLike) -> ComplexMatrix:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("matrix contains non-finite values")
    return arr


def entry_phase_family(matrix: ArrayLike, row: int, col: int) -> EntryPhaseFamily:
    arr = _as_square_complex(matrix)
    n = arr.shape[0]
    if not (0 <= row < n and 0 <= col < n):
        raise IndexError("entry index out of bounds")
    value = arr[row, col]
    return EntryPhaseFamily(
        matrix=arr,
        row=row,
        col=col,
        baseline_phase=float(np.angle(value)),
        magnitude=float(np.abs(value)),
    )


def phase_rotate_entry(matrix: ArrayLike, row: int, col: int, phase_offset: float) -> ComplexMatrix:
    """Rotate one entry by ``phase_offset`` without changing its magnitude.

    The baseline matrix is recovered at phase_offset = 0. Zero entries remain zero.
    """

    return entry_phase_family(matrix, row, col).at(phase_offset)


def phase_sweep_entry(
    matrix: ArrayLike,
    row: int,
    col: int,
    phases: Iterable[float],
) -> list[ComplexMatrix]:
    family = entry_phase_family(matrix, row, col)
    return [family.at(float(phi)) for phi in phases]


def real_pair_rotation(
    matrix: ArrayLike,
    first: tuple[int, int],
    second: tuple[int, int],
    angle: float,
) -> ComplexMatrix:
    """Rotate two real-valued couplings while preserving their joint Euclidean norm.

    This is a physically realizable alternative to complex phase rotation for real
    optimizer Jacobians. The two selected entries are treated as a 2-vector.
    """

    arr = _as_square_complex(matrix)
    if np.max(np.abs(arr.imag)) > 1e-12:
        raise ValueError("real_pair_rotation expects a real matrix")
    n = arr.shape[0]
    for row, col in (first, second):
        if not (0 <= row < n and 0 <= col < n):
            raise IndexError("entry index out of bounds")
    x = float(arr[first].real)
    y = float(arr[second].real)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = arr.copy()
    out[first] = c * x - s * y
    out[second] = s * x + c * y
    return out


def rotate_block_svd_phase(
    matrix: ArrayLike,
    rows: slice | list[int] | NDArray[np.integer],
    cols: slice | list[int] | NDArray[np.integer],
    phase_offsets: ArrayLike,
) -> ComplexMatrix:
    """Rotate singular channels of a coupling block while preserving singular values.

    If B = U diag(s) V*, this returns U diag(s exp(i phi)) V*.  The singular
    values, Frobenius norm, and operator norm of the selected block are unchanged.
    """

    arr = _as_square_complex(matrix)
    out = arr.copy()
    block = out[rows, cols]
    if block.ndim != 2:
        raise ValueError("selected block must be two-dimensional")
    u, singular_values, vh = np.linalg.svd(block, full_matrices=False)
    phases = np.asarray(phase_offsets, dtype=float)
    if phases.ndim == 0:
        phases = np.full_like(singular_values, float(phases))
    if phases.shape != singular_values.shape:
        raise ValueError(
            f"phase_offsets must be scalar or shape {singular_values.shape}, got {phases.shape}"
        )
    rotated = (u * (singular_values * np.exp(1j * phases))) @ vh
    out[rows, cols] = rotated
    return out


def phase_perturbation_norm(entry_value: complex, phase_offset: float) -> float:
    """Exact spectral/Frobenius norm of a one-entry phase perturbation."""

    return float(2.0 * abs(entry_value) * abs(np.sin(phase_offset / 2.0)))
