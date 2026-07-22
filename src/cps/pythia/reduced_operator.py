from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from .basis import BasisVector


@dataclass(frozen=True)
class ProjectionDiagnostics:
    rank: int
    maximum_closure_residual: float
    mean_closure_residual: float
    jvp_norms: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def project_jacobian(
    jvp: Callable[[object], object],
    basis: Sequence[BasisVector],
) -> tuple[np.ndarray, ProjectionDiagnostics]:
    if not basis:
        raise ValueError("basis must not be empty")
    rank = len(basis)
    reduced = np.empty((rank, rank), dtype=np.complex128)
    residuals: list[float] = []
    norms: list[float] = []
    for column, source in enumerate(basis):
        image = jvp(source.vector)
        norm = float(image.norm())
        norms.append(norm)
        projected = image.new_zeros(image.shape)
        for row, target in enumerate(basis):
            coefficient = target.vector.conj() @ image
            reduced[row, column] = complex(coefficient.detach().cpu().item())
            projected = projected + coefficient * target.vector
        residual = float((image - projected).norm()) / max(norm, 1e-30)
        residuals.append(residual)
    diagnostics = ProjectionDiagnostics(
        rank=rank,
        maximum_closure_residual=max(residuals),
        mean_closure_residual=float(np.mean(residuals)),
        jvp_norms=tuple(norms),
    )
    return reduced, diagnostics
