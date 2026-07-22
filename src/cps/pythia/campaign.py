from __future__ import annotations

import json
from pathlib import Path

from .features import discover_feature_records, write_feature_table
from .prediction import evaluate_incremental_value
from .variance import decompose_step_seed_variance


def aggregate_campaign(artifact_root: str | Path, output_csv: str | Path) -> Path:
    records = discover_feature_records(artifact_root)
    if not records:
        raise ValueError(f"no CPS manifests discovered under {artifact_root}")
    return write_feature_table(records, output_csv)


def evaluate_campaign(
    csv_path: str | Path,
    *,
    label_column: str,
    group_column: str,
    output_json: str | Path,
) -> Path:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    baseline = [column for column in ("loss", "gradient_norm", "update_norm") if column in frame]
    if not baseline:
        raise ValueError("feature table needs at least one conventional baseline column")
    cps = [column for column in frame if column.startswith("cps_")]
    result = evaluate_incremental_value(
        frame,
        label_column=label_column,
        group_column=group_column,
        baseline_columns=baseline,
        cps_columns=cps,
    )
    payload: dict[str, object] = {"prediction": result.to_dict()}
    if "step" in frame and "seed" in frame:
        payload["variance"] = {
            column: decompose_step_seed_variance(frame, column).to_dict() for column in cps
        }
    target = Path(output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
