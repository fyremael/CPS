from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from .spectra import spectral_sweep


ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class CandidateScore:
    value: float
    risk: float
    spectral_radius: float
    transient_gain: float
    minimum_gap: float


def score_candidate(
    matrix: ComplexMatrix,
    edges: Iterable[tuple[int, int]],
    phase_count: int = 33,
    horizon: int = 12,
) -> CandidateScore:
    phases = np.linspace(0.0, 2.0 * np.pi, phase_count)
    radius = 0.0
    gain = 1.0
    gap = float("inf")
    for row, col in edges:
        result = spectral_sweep(
            matrix,
            row,
            col,
            phases=phases,
            finite_horizon=horizon,
            compute_kreiss=False,
        )
        radius = max(radius, result.metrics.spectral_radius_max)
        gain = max(gain, result.metrics.finite_horizon_gain)
        gap = min(gap, result.metrics.minimum_gap)
    risk = radius + 0.15 * np.log1p(gain) + 0.01 / max(gap, 1e-12)
    return CandidateScore(
        value=float("nan"),
        risk=float(risk),
        spectral_radius=float(radius),
        transient_gain=float(gain),
        minimum_gap=float(gap),
    )


def select_hyperparameter(
    candidates: Iterable[float],
    operator_builder: Callable[[float], ComplexMatrix],
    edges: Iterable[tuple[int, int]],
) -> CandidateScore:
    """Select a scalar hyperparameter by minimum CPS risk."""

    best: CandidateScore | None = None
    for value in candidates:
        score = score_candidate(operator_builder(float(value)), edges)
        score = CandidateScore(value=float(value), **{k: v for k, v in score.__dict__.items() if k != "value"})
        if best is None or score.risk < best.risk:
            best = score
    if best is None:
        raise ValueError("candidates must not be empty")
    return best
