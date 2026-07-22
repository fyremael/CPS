from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from cps.controllers import CandidateScore, select_hyperparameter


@dataclass(frozen=True)
class PlannerRecommendation:
    control: str
    baseline: float
    recommended: float
    baseline_risk: float
    recommended_risk: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_scalar_control(
    control: str,
    baseline: float,
    candidates: Iterable[float],
    operator_builder,
    edges: Iterable[tuple[int, int]],
) -> PlannerRecommendation:
    baseline_score: CandidateScore = select_hyperparameter([baseline], operator_builder, edges)
    best = select_hyperparameter(candidates, operator_builder, edges)
    direction = "lower" if best.value < baseline else "higher"
    return PlannerRecommendation(
        control=control,
        baseline=float(baseline),
        recommended=float(best.value),
        baseline_risk=float(baseline_score.risk),
        recommended_risk=float(best.risk),
        rationale=(
            f"The {direction} candidate minimized the preregistered CPS risk score over the "
            "candidate grid. Treat this as a continuation-run hypothesis, not as proof."
        ),
    )


def damping_family(reduced_operator: np.ndarray, gamma: float) -> np.ndarray:
    """Simple isotropic contraction used for first-pass damping planning.

    This abstract family scales the entire reduced transition operator toward
    zero. Real optimizer interventions must be tested through the matched
    continuation harness rather than identified with this surrogate.
    """

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    return (1.0 - gamma) * reduced_operator
