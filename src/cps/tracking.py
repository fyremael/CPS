from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]


def _normalize_columns(vectors: ComplexMatrix) -> ComplexMatrix:
    norms = np.linalg.norm(vectors, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return vectors / norms


def match_eigenpairs(
    previous_values: ComplexVector,
    previous_vectors: ComplexMatrix,
    current_values: ComplexVector,
    current_vectors: ComplexMatrix,
    value_weight: float = 0.35,
    overlap_weight: float = 0.65,
) -> NDArray[np.int64]:
    """Match current eigenpairs to previous eigenpairs using assignment.

    Eigenvector overlap carries most of the weight; normalized eigenvalue distance
    stabilizes matching when eigenvectors become unreliable near degeneracy.
    """

    if previous_values.shape != current_values.shape:
        raise ValueError("eigenvalue arrays must have the same shape")
    pvec = _normalize_columns(previous_vectors)
    cvec = _normalize_columns(current_vectors)
    overlap_cost = 1.0 - np.abs(pvec.conj().T @ cvec)

    scale = max(
        float(np.max(np.abs(previous_values))),
        float(np.max(np.abs(current_values))),
        1.0,
    )
    value_cost = np.abs(previous_values[:, None] - current_values[None, :]) / scale
    cost = overlap_weight * overlap_cost + value_weight * value_cost
    rows, cols = linear_sum_assignment(cost)
    order = np.empty_like(cols)
    order[rows] = cols
    return order.astype(np.int64)


def track_eigenpairs(
    eigenvalues: list[ComplexVector],
    eigenvectors: list[ComplexMatrix],
) -> tuple[ComplexMatrix, list[ComplexMatrix]]:
    if len(eigenvalues) != len(eigenvectors):
        raise ValueError("eigenvalue and eigenvector sequences must have equal length")
    if not eigenvalues:
        raise ValueError("cannot track an empty sequence")

    tracked_values = [np.asarray(eigenvalues[0], dtype=np.complex128)]
    tracked_vectors = [np.asarray(eigenvectors[0], dtype=np.complex128)]
    for values, vectors in zip(eigenvalues[1:], eigenvectors[1:]):
        values = np.asarray(values, dtype=np.complex128)
        vectors = np.asarray(vectors, dtype=np.complex128)
        order = match_eigenpairs(
            tracked_values[-1], tracked_vectors[-1], values, vectors
        )
        tracked_values.append(values[order])
        tracked_vectors.append(vectors[:, order])
    return np.stack(tracked_values, axis=0), tracked_vectors
