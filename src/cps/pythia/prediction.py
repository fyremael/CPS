from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProspectivePredictionResult:
    baseline_auc_mean: float
    augmented_auc_mean: float
    delta_auc: float
    folds: int
    baseline_columns: tuple[str, ...]
    cps_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_incremental_value(
    frame,
    *,
    label_column: str,
    group_column: str,
    baseline_columns: Sequence[str],
    cps_columns: Sequence[str],
) -> ProspectivePredictionResult:
    """Leave-one-group-out comparison of baseline and baseline+CPS predictors."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra for prediction evaluation") from exc

    baseline_scores: list[float] = []
    augmented_scores: list[float] = []
    groups = list(frame[group_column].dropna().unique())
    for group in groups:
        train = frame[frame[group_column] != group]
        test = frame[frame[group_column] == group]
        if train[label_column].nunique() < 2 or test[label_column].nunique() < 2:
            continue

        def fit_score(columns: Sequence[str]) -> float:
            pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
            )
            pipeline.fit(train[list(columns)], train[label_column])
            probability = pipeline.predict_proba(test[list(columns)])[:, 1]
            return float(roc_auc_score(test[label_column], probability))

        baseline_scores.append(fit_score(baseline_columns))
        augmented_scores.append(fit_score((*baseline_columns, *cps_columns)))

    if not baseline_scores:
        raise ValueError("no valid leave-one-group-out folds contained both classes")
    baseline = float(np.mean(baseline_scores))
    augmented = float(np.mean(augmented_scores))
    return ProspectivePredictionResult(
        baseline_auc_mean=baseline,
        augmented_auc_mean=augmented,
        delta_auc=augmented - baseline,
        folds=len(baseline_scores),
        baseline_columns=tuple(baseline_columns),
        cps_columns=tuple(cps_columns),
    )
