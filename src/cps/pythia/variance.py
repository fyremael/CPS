from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceDecomposition:
    total_variance: float
    between_step_variance: float
    between_seed_variance: float
    residual_variance: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def decompose_step_seed_variance(frame, value_column: str) -> VarianceDecomposition:
    values = frame[value_column].astype(float)
    total = float(values.var(ddof=1))
    grand = float(values.mean())
    step_means = frame.groupby("step")[value_column].mean()
    seed_means = frame.groupby("seed")[value_column].mean()
    step_variance = float(np.mean((step_means.to_numpy(dtype=float) - grand) ** 2))
    seed_variance = float(np.mean((seed_means.to_numpy(dtype=float) - grand) ** 2))
    residual = max(0.0, total - step_variance - seed_variance)
    return VarianceDecomposition(total, step_variance, seed_variance, residual)
