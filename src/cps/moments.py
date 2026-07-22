from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .perturbations import phase_sweep_entry


ComplexMatrix = NDArray[np.complex128]


def trace_moment(matrix: ArrayLike, order: int) -> complex:
    if order < 1:
        raise ValueError("order must be positive")
    a = np.asarray(matrix, dtype=np.complex128)
    return complex(np.trace(np.linalg.matrix_power(a, order)))


def phase_moment_spectrum(
    matrix: ArrayLike,
    row: int,
    col: int,
    max_order: int,
    phases: ArrayLike | None = None,
) -> dict[int, NDArray[np.complex128]]:
    """Fourier spectra of trace moments along an entry-phase sweep.

    Harmonics reveal how often the selected coupling participates in closed walks
    contributing to tr(A^q). The result is invariant to eigenvalue branch labeling.
    """

    if max_order < 1:
        raise ValueError("max_order must be positive")
    if phases is None:
        count = max(64, 4 * max_order + 1)
        phases = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    phase_array = np.asarray(phases, dtype=float)
    matrices = phase_sweep_entry(matrix, row, col, phase_array)
    spectra: dict[int, NDArray[np.complex128]] = {}
    for order in range(1, max_order + 1):
        samples = np.asarray([trace_moment(a, order) for a in matrices])
        spectra[order] = np.fft.fft(samples) / samples.size
    return spectra
