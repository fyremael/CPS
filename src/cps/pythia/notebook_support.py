from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import load_probe_config
from .planner import PlannerRecommendation
from .runner import run_probe


@dataclass(frozen=True)
class EvidencePacket:
    root: Path
    reduced_operator_path: Path
    matrix: np.ndarray
    couplings_path: Path | None
    manifest_path: Path | None


def build_probe_for_notebook(
    *,
    revision: str | None = None,
    run_name: str,
    root: str = "/content/cps-artifacts",
    config_path: str = "subjects/pythia/configs/pythia_70m_smoke.yaml",
):
    """Build the compact, robust probe used by self-contained release notebooks."""

    config = load_probe_config(config_path)
    if revision is not None:
        config = replace(config, model=replace(config.model, revision=revision))
    return replace(
        config,
        jacobian=replace(config.jacobian, mode="finite_difference"),
        output=replace(config.output, root=root, run_name=run_name),
    )


def run_self_contained_probe(
    *,
    revision: str | None = None,
    run_name: str,
    root: str = "/content/cps-artifacts",
) -> Path:
    config = build_probe_for_notebook(revision=revision, run_name=run_name, root=root)
    return run_probe(config)


def load_evidence_packet(path: str | Path) -> EvidencePacket:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        destination = Path("/content/cps-import") / candidate.stem.replace(" ", "-")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(candidate), str(destination), "zip")
        candidate = destination
    if candidate.is_file():
        if candidate.name != "reduced_operator.npy":
            raise FileNotFoundError(
                f"expected reduced_operator.npy, a ZIP archive, or an evidence directory; got {candidate}"
            )
        root = candidate.parent
    else:
        root = candidate
    if not root.exists():
        raise FileNotFoundError(f"evidence path does not exist: {root}")
    operator = root / "reduced_operator.npy"
    if not operator.is_file():
        matches = sorted(root.rglob("reduced_operator.npy"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"could not resolve a unique reduced_operator.npy under {root}; found {len(matches)}"
            )
        operator = matches[0]
        root = operator.parent
    couplings = root / "couplings.json"
    manifest = root / "manifest.json"
    return EvidencePacket(
        root=root,
        reduced_operator_path=operator,
        matrix=np.load(operator),
        couplings_path=couplings if couplings.is_file() else None,
        manifest_path=manifest if manifest.is_file() else None,
    )


def select_candidate_edges(packet: EvidencePacket, maximum: int = 8) -> list[tuple[int, int]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if packet.couplings_path is not None:
        records = json.loads(packet.couplings_path.read_text(encoding="utf-8"))
        if records:
            return [
                (int(record["row"]), int(record["col"]))
                for record in records[:maximum]
            ]
    magnitude = np.abs(packet.matrix).copy()
    np.fill_diagonal(magnitude, 0.0)
    flat = np.argsort(magnitude.ravel())[::-1]
    edges: list[tuple[int, int]] = []
    for index in flat:
        row, col = np.unravel_index(index, magnitude.shape)
        if magnitude[row, col] <= 0:
            break
        edges.append((int(row), int(col)))
        if len(edges) >= maximum:
            break
    return edges


def write_planner_recommendation(
    root: str | Path,
    recommendation: PlannerRecommendation,
    *,
    selected_edges: Iterable[tuple[int, int]],
    candidate_grid: Iterable[float],
    control_operationalization: dict[str, object],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "planner_recommendation.json"
    payload = recommendation.to_dict()
    payload["selected_edges"] = [list(edge) for edge in selected_edges]
    payload["candidate_grid"] = [float(value) for value in candidate_grid]
    payload["control_operationalization"] = control_operationalization
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


__all__ = [
    "EvidencePacket",
    "build_probe_for_notebook",
    "load_evidence_packet",
    "run_self_contained_probe",
    "select_candidate_edges",
    "write_planner_recommendation",
]
