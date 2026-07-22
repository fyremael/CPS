from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CheckpointFeatures:
    artifact_root: str
    run: str
    revision: str
    step: int
    seed: int | None
    selected_parameter_numel: int
    basis_rank: int
    closure_residual_max: float
    cps_spectral_radius_max: float
    cps_transient_gain_max: float
    cps_minimum_gap_min: float
    cps_eigenvalue_condition_max: float
    cps_loop_length_max: float
    cps_displacement_max: float
    coupling_count: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def extract_checkpoint_features(root: str | Path) -> CheckpointFeatures:
    path = Path(root)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    couplings = json.loads((path / "couplings.json").read_text(encoding="utf-8"))
    if not couplings:
        raise ValueError(f"no coupling records under {path}")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(item["metrics"][key]) for item in couplings], dtype=float)

    run_spec = manifest.get("run_spec", {})
    revision = str(manifest["revision"])
    return CheckpointFeatures(
        artifact_root=str(path),
        run=str(run_spec.get("key", manifest.get("model_id", "unknown"))),
        revision=revision,
        step=int(revision.removeprefix("step")),
        seed=run_spec.get("seed"),
        selected_parameter_numel=int(manifest["selected_parameter_numel"]),
        basis_rank=int(manifest["basis_rank"]),
        closure_residual_max=float(manifest["projection"]["maximum_closure_residual"]),
        cps_spectral_radius_max=float(np.nanmax(values("spectral_radius_max"))),
        cps_transient_gain_max=float(np.nanmax(values("finite_horizon_gain"))),
        cps_minimum_gap_min=float(np.nanmin(values("minimum_gap"))),
        cps_eigenvalue_condition_max=float(
            np.nanmax(values("maximum_eigenvalue_condition"))
        ),
        cps_loop_length_max=float(np.nanmax(values("total_loop_length"))),
        cps_displacement_max=float(np.nanmax(values("baseline_displacement_max"))),
        coupling_count=len(couplings),
    )


def discover_feature_records(root: str | Path) -> tuple[CheckpointFeatures, ...]:
    records: list[CheckpointFeatures] = []
    for manifest in sorted(Path(root).rglob("manifest.json")):
        try:
            records.append(extract_checkpoint_features(manifest.parent))
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return tuple(records)


def write_feature_table(records: Iterable[CheckpointFeatures], output: str | Path) -> Path:
    import pandas as pd

    frame = pd.DataFrame([record.to_dict() for record in records])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
