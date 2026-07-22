from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cps.families import singular_channel_family, sweep_matrix_family

from .basis import BasisVector


@dataclass(frozen=True)
class CouplingRecord:
    row: int
    col: int
    source: str
    target: str
    magnitude: float
    metrics: dict[str, float]
    family_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_couplings(matrix: np.ndarray, maximum: int) -> tuple[tuple[int, int], ...]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    magnitude = np.abs(matrix).copy()
    np.fill_diagonal(magnitude, 0.0)
    flat = np.argsort(magnitude.ravel())[::-1]
    output: list[tuple[int, int]] = []
    for index in flat:
        row, col = np.unravel_index(index, magnitude.shape)
        if magnitude[row, col] <= 0:
            break
        output.append((int(row), int(col)))
        if len(output) >= maximum:
            break
    return tuple(output)


def analyze_reduced_operator(
    matrix: np.ndarray,
    basis: Sequence[BasisVector],
    *,
    phase_count: int,
    finite_horizon: int,
    compute_kreiss: bool,
    maximum_couplings: int,
    progress: Callable[[int, int, CouplingRecord], None] | None = None,
) -> tuple[CouplingRecord, ...]:
    phases = np.linspace(0.0, 2.0 * np.pi, phase_count)
    records: list[CouplingRecord] = []
    selected = select_couplings(matrix, maximum_couplings)
    for row, col in selected:
        family, metadata = singular_channel_family(matrix, [row], [col], channel=0)
        sweep = sweep_matrix_family(
            family,
            phases,
            family_name="reduced-entry-singular-channel",
            finite_horizon=finite_horizon,
            compute_kreiss=compute_kreiss,
            metadata=metadata,
        )
        record = CouplingRecord(
            row=row,
            col=col,
            source=basis[col].name,
            target=basis[row].name,
            magnitude=float(abs(matrix[row, col])),
            metrics=sweep.metrics.to_dict(),
            family_metadata=metadata,
        )
        records.append(record)
        if progress is not None:
            progress(len(records), len(selected), record)
    return tuple(records)


def save_analysis(
    output_dir: str | Path,
    matrix: np.ndarray,
    basis: Sequence[BasisVector],
    records: Sequence[CouplingRecord],
    manifest: dict[str, object],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "reduced_operator.npy", matrix)
    (root / "basis.json").write_text(
        json.dumps(
            [
                {
                    "index": index,
                    "name": item.name,
                    "parameter_name": item.parameter_name,
                    "component": item.component,
                }
                for index, item in enumerate(basis)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "couplings.json").write_text(
        json.dumps([record.to_dict() for record in records], indent=2), encoding="utf-8"
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
