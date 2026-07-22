from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64] | NDArray[np.complex128]
Matrix = NDArray[np.float64] | NDArray[np.complex128]
MatVec = Callable[[Vector], Vector]


def project_dense_operator(matrix: ArrayLike, basis: ArrayLike) -> Matrix:
    a = np.asarray(matrix)
    q = np.asarray(basis)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if q.ndim != 2 or q.shape[0] != a.shape[0]:
        raise ValueError("basis shape is incompatible with matrix")
    gram = q.conj().T @ q
    if not np.allclose(gram, np.eye(q.shape[1]), atol=1e-7):
        raise ValueError("basis columns must be orthonormal")
    return q.conj().T @ a @ q


def arnoldi_projection(
    matvec: MatVec,
    dimension: int,
    rank: int,
    initial: ArrayLike | None = None,
    tol: float = 1e-12,
) -> tuple[Matrix, Matrix]:
    """Build an Arnoldi basis and reduced Hessenberg operator using only matvecs."""

    if not (1 <= rank <= dimension):
        raise ValueError("rank must lie between 1 and dimension")
    if initial is None:
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(dimension)
    else:
        vector = np.asarray(initial)
        if vector.shape != (dimension,):
            raise ValueError("initial vector has wrong shape")
    dtype = np.result_type(vector.dtype, np.complex128)
    q = np.zeros((dimension, rank + 1), dtype=dtype)
    h = np.zeros((rank + 1, rank), dtype=dtype)
    norm = np.linalg.norm(vector)
    if norm <= tol:
        raise ValueError("initial vector must be nonzero")
    q[:, 0] = vector / norm

    actual_rank = rank
    for j in range(rank):
        w = np.asarray(matvec(q[:, j]), dtype=dtype)
        if w.shape != (dimension,):
            raise ValueError("matvec returned a vector with the wrong shape")
        for i in range(j + 1):
            h[i, j] = np.vdot(q[:, i], w)
            w = w - h[i, j] * q[:, i]
        h[j + 1, j] = np.linalg.norm(w)
        if h[j + 1, j] <= tol:
            actual_rank = j + 1
            break
        q[:, j + 1] = w / h[j + 1, j]
    basis = q[:, :actual_rank]
    reduced = h[:actual_rank, :actual_rank]
    return basis, reduced


def randomized_projection(
    matvec: MatVec,
    dimension: int,
    rank: int,
    oversampling: int = 8,
    seed: int = 0,
) -> tuple[Matrix, Matrix]:
    """Randomized range projection for a matrix-free linear operator."""

    if rank < 1 or rank > dimension:
        raise ValueError("rank must lie between 1 and dimension")
    width = min(dimension, rank + max(0, oversampling))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dimension, width))
    samples = np.column_stack([matvec(omega[:, j]) for j in range(width)])
    q, _ = np.linalg.qr(samples, mode="reduced")
    q = q[:, :rank]
    aq = np.column_stack([matvec(q[:, j]) for j in range(rank)])
    reduced = q.conj().T @ aq
    return q, reduced
