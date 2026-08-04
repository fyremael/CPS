from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cps.metrics import eigenvalue_condition_numbers
from cps.subspace_stability import (
    SubspaceStabilityCertificate,
    leading_singular_subspace_stability,
)

from .fixtures import RegimeFixture, regime_fixtures, rotation

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("artifacts/tran_vu_characterization")
EPSILON = 1e-12


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: ";".join(str(item) for item in value)
                if isinstance(value, (tuple, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def _expectation_errors(
    fixture: RegimeFixture,
    certificate: SubspaceStabilityCertificate,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_flags = {
        "theorem_applicable": fixture.expected_theorem_applicable,
        "informative": fixture.expected_informative,
        "improves_classical": fixture.expected_improves_classical,
        "admitted": fixture.expected_admitted,
    }
    for field, expected in expected_flags.items():
        actual = bool(getattr(certificate, field))
        if actual != expected:
            errors.append(f"{field}: expected {expected}, observed {actual}")
    if fixture.expected_reason is not None and fixture.expected_reason not in certificate.reasons:
        errors.append(
            f"required reason {fixture.expected_reason!r} absent from {certificate.reasons!r}"
        )
    if (
        fixture.expected_representation is not None
        and certificate.representation != fixture.expected_representation
    ):
        errors.append(
            "representation: expected "
            f"{fixture.expected_representation!r}, observed {certificate.representation!r}"
        )
    if fixture.minimum_halving_rank is not None:
        if certificate.halving_rank is None or certificate.halving_rank < fixture.minimum_halving_rank:
            errors.append(
                "halving_rank: expected at least "
                f"{fixture.minimum_halving_rank}, observed {certificate.halving_rank}"
            )
    return tuple(errors)


def _nonnormality_score(matrix: np.ndarray) -> float:
    operator_norm = float(np.linalg.norm(matrix, ord=2))
    if operator_norm == 0.0:
        return 0.0
    commutator = matrix @ matrix.conj().T - matrix.conj().T @ matrix
    return float(np.linalg.norm(commutator, ord=2) / operator_norm**2)


def _maximum_eigenvalue_condition(matrix: np.ndarray) -> float:
    values = eigenvalue_condition_numbers(matrix)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.max(finite))


def _regime_row(fixture: RegimeFixture) -> dict[str, object]:
    certificate = leading_singular_subspace_stability(
        fixture.reference,
        fixture.perturbed,
        rank=fixture.rank,
    )
    errors = _expectation_errors(fixture, certificate)
    row = {
        "name": fixture.name,
        "title": fixture.title,
        "description": fixture.description,
        "rank": fixture.rank,
        "tags": fixture.tags,
        "expectation_passed": not errors,
        "expectation_errors": errors,
        "nonnormality_score": _nonnormality_score(fixture.reference),
        "maximum_eigenvalue_condition": _maximum_eigenvalue_condition(fixture.reference),
    }
    row.update(certificate.to_dict())
    return row


def _sweep_row(
    parameter_name: str,
    parameter: float,
    certificate: SubspaceStabilityCertificate,
) -> dict[str, object]:
    row: dict[str, object] = {parameter_name: float(parameter)}
    row.update(certificate.to_dict())
    return row


def _directional_coupling_sweep(points: int) -> list[dict[str, object]]:
    reference = np.diag([1000.0, 996.0, 0.0]).astype(np.complex128)
    noise = 0.5
    rows: list[dict[str, object]] = []
    for ratio in np.linspace(0.0, 1.0, points):
        local = noise * float(ratio)
        remote = math.sqrt(max(noise**2 - local**2, 0.0))
        perturbation = np.zeros_like(reference)
        perturbation[0, 1] = local
        perturbation[0, 2] = remote
        certificate = leading_singular_subspace_stability(reference, reference + perturbation)
        row = _sweep_row("directional_coupling_ratio", float(ratio), certificate)
        row["configured_perturbation_norm"] = noise
        rows.append(row)
    return rows


def _gap_sweep(points: int) -> list[dict[str, object]]:
    noise = 0.5
    rows: list[dict[str, object]] = []
    for gap in np.geomspace(1.25, 300.0, points):
        reference = np.diag([1000.0, 1000.0 - float(gap), 0.0]).astype(np.complex128)
        perturbation = np.zeros_like(reference)
        perturbation[0, 2] = noise
        certificate = leading_singular_subspace_stability(reference, reference + perturbation)
        row = _sweep_row("configured_gap", float(gap), certificate)
        row["gap_to_noise_ratio"] = float(gap / noise)
        rows.append(row)
    return rows


def _perturbation_sweep(points: int) -> list[dict[str, object]]:
    reference = np.diag([1000.0, 996.0, 0.0]).astype(np.complex128)
    rows: list[dict[str, object]] = []
    for magnitude in np.geomspace(0.02, 1.5, points):
        perturbation = np.zeros_like(reference)
        perturbation[0, 2] = float(magnitude)
        certificate = leading_singular_subspace_stability(reference, reference + perturbation)
        row = _sweep_row("configured_perturbation_norm", float(magnitude), certificate)
        row["perturbation_to_gap_ratio"] = float(magnitude / 4.0)
        rows.append(row)
    return rows


def _halving_rank_sweep(maximum_rank: int = 8) -> list[dict[str, object]]:
    noise = 0.5
    local = 0.01
    remote = math.sqrt(noise**2 - local**2)
    rows: list[dict[str, object]] = []
    for desired_rank in range(2, maximum_rank + 1):
        nearby = [996.0 - index for index in range(1, desired_rank - 1)]
        singular_values = [1000.0, 996.0, *nearby, 0.0]
        reference = np.diag(singular_values).astype(np.complex128)
        perturbation = np.zeros_like(reference)
        perturbation[0, 1] = local
        perturbation[0, -1] = remote
        certificate = leading_singular_subspace_stability(reference, reference + perturbation)
        row = _sweep_row("configured_halving_rank", float(desired_rank), certificate)
        row["observed_halving_rank"] = certificate.halving_rank
        rows.append(row)
    return rows


def _admission_map(
    gap_points: int,
    coupling_points: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    noise = 0.5
    gap_ratios = np.geomspace(2.5, 100.0, gap_points)
    coupling_ratios = np.linspace(0.0, 0.1, coupling_points)
    admitted = np.zeros((coupling_points, gap_points), dtype=float)
    rows: list[dict[str, object]] = []
    for row_index, coupling_ratio in enumerate(coupling_ratios):
        local = noise * float(coupling_ratio)
        remote = math.sqrt(max(noise**2 - local**2, 0.0))
        for column_index, gap_ratio in enumerate(gap_ratios):
            gap = noise * float(gap_ratio)
            reference = np.diag([1000.0, 1000.0 - gap, 0.0]).astype(np.complex128)
            perturbation = np.zeros_like(reference)
            perturbation[0, 1] = local
            perturbation[0, 2] = remote
            certificate = leading_singular_subspace_stability(reference, reference + perturbation)
            admitted[row_index, column_index] = 1.0 if certificate.admitted else 0.0
            rows.append(
                {
                    "gap_to_noise_ratio": float(gap_ratio),
                    "directional_coupling_ratio": float(coupling_ratio),
                    "admitted": certificate.admitted,
                    "theorem_applicable": certificate.theorem_applicable,
                    "informative": certificate.informative,
                    "improves_classical": certificate.improves_classical,
                    "davis_kahan_bound": certificate.davis_kahan_bound,
                    "tran_vu_bound": certificate.tran_vu_bound,
                    "observed_projector_distance": certificate.observed_projector_distance,
                }
            )
    return rows, gap_ratios, coupling_ratios, admitted


def _nonnormality_sweep(points: int) -> list[dict[str, object]]:
    singular_values = np.diag([1000.0, 996.0, 0.0]).astype(np.complex128)
    rows: list[dict[str, object]] = []
    for angle in np.linspace(0.0, 1.2, points):
        left = rotation(3, 0, 1, float(angle))
        right = rotation(3, 1, 2, float(-0.75 * angle))
        reference = left @ singular_values @ right.conj().T
        perturbation = 0.5 * np.outer(left[:, 0], right[:, 2].conj())
        certificate = leading_singular_subspace_stability(reference, reference + perturbation)
        rows.append(
            {
                "angle": float(angle),
                "nonnormality_score": _nonnormality_score(reference),
                "maximum_eigenvalue_condition": _maximum_eigenvalue_condition(reference),
                "observed_projector_distance": certificate.observed_projector_distance,
                "davis_kahan_bound": certificate.davis_kahan_bound,
                "tran_vu_bound": certificate.tran_vu_bound,
                "admitted": certificate.admitted,
            }
        )
    return rows


def _realification(matrix: np.ndarray) -> np.ndarray:
    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


def _realification_validation(
    samples: int = 12,
    seed: int = 20260727,
) -> list[dict[str, object]]:
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(samples):
        matrix = generator.normal(size=(4, 3)) + 1j * generator.normal(size=(4, 3))
        realified = _realification(matrix)
        complex_norm = float(np.linalg.norm(matrix, ord=2))
        real_norm = float(np.linalg.norm(realified, ord=2))
        complex_singular = np.linalg.svd(matrix, compute_uv=False)
        real_singular = np.linalg.svd(realified, compute_uv=False)
        duplicated = np.sort(np.repeat(complex_singular, 2))[::-1]
        singular_error = float(
            np.linalg.norm(real_singular - duplicated, ord=np.inf)
            / max(float(np.linalg.norm(duplicated, ord=np.inf)), EPSILON)
        )
        rows.append(
            {
                "sample": index,
                "operator_norm_relative_error": abs(real_norm - complex_norm)
                / max(complex_norm, EPSILON),
                "singular_duplication_relative_error": singular_error,
            }
        )
    return rows


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_dir / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _positive(value: object) -> float:
    if value is None:
        return float("nan")
    numeric = float(value)
    return max(numeric, EPSILON)


def _plot_regime_bounds(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    positions = np.arange(len(rows), dtype=float)
    width = 0.25
    observed = [_positive(row["observed_projector_distance"]) for row in rows]
    classical = [_positive(row["davis_kahan_bound"]) for row in rows]
    refined = [_positive(row["tran_vu_bound"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(positions - width, observed, width, label="Observed projector motion")
    ax.bar(positions, classical, width, label="Davis--Kahan bound")
    ax.bar(positions + width, refined, width, label="Tran--Vu bound")
    ax.set_yscale("log")
    ax.set_ylabel("Distance or upper bound")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(row["title"]) for row in rows], rotation=32, ha="right")
    ax.set_title("Governed regime matrix: observed motion and certified bounds")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir, "01_regime_bound_comparison")


def _plot_directional_coupling(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    x = [float(row["directional_coupling_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(x, [_positive(row["tran_vu_bound"]) for row in rows], label="Tran--Vu bound")
    ax.plot(x, [_positive(row["davis_kahan_bound"]) for row in rows], label="Davis--Kahan bound")
    ax.plot(
        x,
        [_positive(row["observed_projector_distance"]) for row in rows],
        label="Observed motion",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Informative-bound threshold")
    ax.set_yscale("log")
    ax.set_xlabel(r"Directional coupling ratio $x/\|E\|_2$")
    ax.set_ylabel("Distance or upper bound")
    ax.set_title("Fixed perturbation norm: local direction determines usefulness")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "02_directional_coupling_sweep")


def _plot_gap_sweep(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    x = [float(row["gap_to_noise_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(x, [_positive(row["tran_vu_bound"]) for row in rows], label="Tran--Vu bound")
    ax.plot(x, [_positive(row["davis_kahan_bound"]) for row in rows], label="Davis--Kahan bound")
    ax.plot(
        x,
        [_positive(row["observed_projector_distance"]) for row in rows],
        label="Observed motion",
    )
    ax.axvline(4.0, linestyle="--", linewidth=1.0, label="Moderate-gap lower boundary")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Gap-to-noise ratio $\delta_p/\|E\|_2$")
    ax.set_ylabel("Distance or upper bound")
    ax.set_title("Gap sweep: Tran--Vu is a moderate-gap tool, not a universal replacement")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "03_gap_sweep")


def _plot_perturbation_sweep(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    x = [float(row["perturbation_to_gap_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(x, [_positive(row["tran_vu_bound"]) for row in rows], label="Tran--Vu bound")
    ax.plot(x, [_positive(row["davis_kahan_bound"]) for row in rows], label="Davis--Kahan bound")
    ax.plot(
        x,
        [_positive(row["observed_projector_distance"]) for row in rows],
        label="Observed motion",
    )
    ax.axvline(0.25, linestyle="--", linewidth=1.0, label="Theorem boundary")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Perturbation-to-gap ratio $\|E\|_2/\delta_p$")
    ax.set_ylabel("Distance or upper bound")
    ax.set_title("Perturbation sweep: certificates fail closed beyond the theorem boundary")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "04_perturbation_sweep")


def _plot_halving_rank(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    x = [float(row["configured_halving_rank"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(
        x,
        [_positive(row["tran_vu_bound"]) for row in rows],
        marker="o",
        label="Tran--Vu bound",
    )
    ax.plot(
        x,
        [_positive(row["davis_kahan_bound"]) for row in rows],
        marker="o",
        label="Davis--Kahan bound",
    )
    ax.plot(
        x,
        [_positive(row["observed_projector_distance"]) for row in rows],
        marker="o",
        label="Observed motion",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Informative-bound threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Halving rank r")
    ax.set_ylabel("Distance or upper bound")
    ax.set_title(r"Cluster breadth exposes the $r^2$ penalty")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "05_halving_rank_penalty")


def _plot_admission_map(
    gap_ratios: np.ndarray,
    coupling_ratios: np.ndarray,
    admitted: np.ndarray,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    image = ax.imshow(
        admitted,
        origin="lower",
        aspect="auto",
        extent=[
            math.log10(float(gap_ratios[0])),
            math.log10(float(gap_ratios[-1])),
            float(coupling_ratios[0]),
            float(coupling_ratios[-1]),
        ],
        interpolation="nearest",
    )
    ticks = [3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
    ax.set_xticks([math.log10(value) for value in ticks])
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.set_xlabel(r"Gap-to-noise ratio $\delta_p/\|E\|_2$")
    ax.set_ylabel(r"Directional coupling ratio $x/\|E\|_2$")
    ax.set_title("Admission map: useful certificates occupy a narrow directional regime")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Not admitted", "Admitted"])
    _save_figure(fig, figure_dir, "06_admission_map")


def _plot_observed_vs_bounds(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    applicable = [row for row in rows if bool(row["theorem_applicable"])]
    observed = np.asarray(
        [_positive(row["observed_projector_distance"]) for row in applicable]
    )
    refined = np.asarray([_positive(row["tran_vu_bound"]) for row in applicable])
    classical = np.asarray([_positive(row["davis_kahan_bound"]) for row in applicable])
    lower = max(min(float(np.min(observed)), EPSILON), EPSILON)
    upper = max(float(np.max(np.concatenate([refined, classical]))), lower * 10.0)
    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    ax.scatter(observed, refined, label="Tran--Vu")
    ax.scatter(observed, classical, marker="x", label="Davis--Kahan")
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.0,
        label="Bound = observed",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed projector motion")
    ax.set_ylabel("Upper bound")
    ax.set_title("Applicable certificates contain the observed subspace displacement")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "07_observed_vs_bounds")


def _plot_nonnormality(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    x = [float(row["nonnormality_score"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        x,
        [_positive(row["observed_projector_distance"]) for row in rows],
        marker="o",
    )
    axes[0].plot(
        x,
        [_positive(row["tran_vu_bound"]) for row in rows],
        label="Tran--Vu bound",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Non-normality score")
    axes[0].set_ylabel("Singular-space distance")
    axes[0].set_title("Certified singular-space behaviour")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x,
        [_positive(row["maximum_eigenvalue_condition"]) for row in rows],
        marker="o",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Non-normality score")
    axes[1].set_ylabel("Maximum eigenvalue condition")
    axes[1].set_title("Eigenvector sensitivity remains separate")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Non-normality separation: singular certificates do not certify eigenvectors")
    _save_figure(fig, figure_dir, "08_nonnormality_separation")


def _plot_realification(
    rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    samples = [int(row["sample"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(
        samples,
        [_positive(row["operator_norm_relative_error"]) for row in rows],
        marker="o",
        label="Operator norm",
    )
    ax.plot(
        samples,
        [_positive(row["singular_duplication_relative_error"]) for row in rows],
        marker="o",
        label="Duplicated singular spectrum",
    )
    ax.axhline(1e-10, linestyle="--", linewidth=1.0, label="Acceptance threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Deterministic complex fixture")
    ax.set_ylabel("Relative error")
    ax.set_title("Complex realification preserves the quantities used by the certificate")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir, "09_realification_validation")


def _format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(numeric):
        return str(numeric)
    if numeric == 0.0:
        return "0"
    if abs(numeric) < 1e-3 or abs(numeric) >= 1e4:
        return f"{numeric:.3e}"
    return f"{numeric:.5g}"


def _html_table(rows: Sequence[dict[str, object]]) -> str:
    columns = [
        ("title", "Regime"),
        ("theorem_applicable", "Applicable"),
        ("admitted", "Admitted"),
        ("gap", "Gap"),
        ("perturbation_norm", "||E||"),
        ("halving_rank", "r"),
        ("directional_coupling_ratio", "x / ||E||"),
        ("observed_projector_distance", "Observed"),
        ("davis_kahan_bound", "Davis--Kahan"),
        ("tran_vu_bound", "Tran--Vu"),
        ("expectation_passed", "Fixture contract"),
    ]
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{_format_number(row.get(key))}</td>" for key, _ in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _write_html_report(
    output_dir: Path,
    regime_rows: Sequence[dict[str, object]],
    acceptance: dict[str, object],
    figure_stems: Sequence[str],
) -> None:
    status = "PASS" if acceptance["passed"] else "FAIL"
    admitted = sum(bool(row["admitted"]) for row in regime_rows)
    applicable = sum(bool(row["theorem_applicable"]) for row in regime_rows)
    figures = "".join(
        (
            '<figure><img src="figures/'
            f"{html.escape(stem)}.png" 
            f'alt="{html.escape(stem.replace("_", " "))}">'
            f"<figcaption>{html.escape(stem.replace('_', ' ').title())}</figcaption></figure>"
        )
        for stem in figure_stems
    )
    gate_rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' '))}</td>"
        f"<td>{'PASS' if detail['passed'] else 'FAIL'}</td>"
        f"<td><code>{html.escape(str(detail.get('value')))}</code></td></tr>"
        for name, detail in acceptance["gates"].items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tran--Vu characterization report</title>
<style>
:root {{ color-scheme: light dark; --accent: #4f6f8f; --surface: rgba(127,127,127,.09); }}
body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1180px; padding: 2rem; line-height: 1.5; }}
h1, h2 {{ line-height: 1.15; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 1rem; margin: 1.5rem 0; }}
.card, figure {{ background: var(--surface); border-radius: 12px; padding: 1rem; }}
.value {{ font-size: 1.8rem; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ padding: .55rem; border-bottom: 1px solid rgba(127,127,127,.25); text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); gap: 1rem; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; display: block; background: white; border-radius: 8px; }}
figcaption {{ margin-top: .6rem; font-weight: 600; }}
.status {{ color: var(--accent); }}
code {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Tran--Vu moderate-gap characterization</h1>
<p>This report characterizes where the refined subspace bound is applicable, informative, and genuinely sharper than classical Davis--Kahan. It is a deterministic synthetic study of the certificate, not evidence about a trained model.</p>
<div class="summary">
<div class="card"><div>Acceptance</div><div class="value status">{status}</div></div>
<div class="card"><div>Governed regimes</div><div class="value">{len(regime_rows)}</div></div>
<div class="card"><div>Theorem applicable</div><div class="value">{applicable}</div></div>
<div class="card"><div>Admitted improvements</div><div class="value">{admitted}</div></div>
</div>
<h2>Regime matrix</h2>
{_html_table(regime_rows)}
<h2>Acceptance gates</h2>
<table><thead><tr><th>Gate</th><th>Status</th><th>Observed</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Visual characterization</h2>
<div class="gallery">{figures}</div>
<h2>Interpretation boundary</h2>
<p>An admitted result certifies the measured leading singular subspace for the declared perturbation. It does not certify non-normal eigenvectors, pseudospectral amplification, projection closure, future training stability, or causal usefulness.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _load_acceptance_spec() -> dict[str, object]:
    return json.loads((PACKAGE_ROOT / "acceptance.json").read_text(encoding="utf-8"))


def _evaluate_acceptance(
    output_dir: Path,
    regime_rows: Sequence[dict[str, object]],
    realification_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    spec = _load_acceptance_spec()
    required_regimes = set(spec["required_regimes"])
    present_regimes = {str(row["name"]) for row in regime_rows}
    required_figures = set(spec["required_figures"])
    present_png = {path.stem for path in (output_dir / "figures").glob("*.png")}
    present_svg = {path.stem for path in (output_dir / "figures").glob("*.svg")}

    applicable = [row for row in regime_rows if bool(row["theorem_applicable"])]
    applicable_cover = all(
        row["tran_vu_bound"] is not None
        and float(row["observed_projector_distance"])
        <= float(row["tran_vu_bound"]) + 1e-10
        for row in applicable
    )
    admitted = [row for row in regime_rows if bool(row["admitted"])]
    admitted_consistent = all(
        bool(row["informative"])
        and bool(row["improves_classical"])
        and float(row["tran_vu_bound"]) < 1.0
        and float(row["tran_vu_bound"]) < float(row["davis_kahan_bound"])
        for row in admitted
    )
    maximum_norm_error = max(
        float(row["operator_norm_relative_error"]) for row in realification_rows
    )
    maximum_singular_error = max(
        float(row["singular_duplication_relative_error"])
        for row in realification_rows
    )
    gate_spec = spec["gates"]
    gates = {
        "all_fixture_expectations_match": {
            "passed": all(bool(row["expectation_passed"]) for row in regime_rows),
            "value": sum(bool(row["expectation_passed"]) for row in regime_rows),
            "required": len(regime_rows),
        },
        "all_applicable_tran_vu_bounds_cover_observed": {
            "passed": applicable_cover,
            "value": len(applicable),
        },
        "all_admitted_results_are_informative_and_sharper": {
            "passed": admitted_consistent,
            "value": len(admitted),
        },
        "all_required_regimes_present": {
            "passed": required_regimes <= present_regimes,
            "value": sorted(present_regimes),
            "missing": sorted(required_regimes - present_regimes),
        },
        "all_required_figures_present": {
            "passed": required_figures <= present_png and required_figures <= present_svg,
            "value": sorted(present_png & present_svg),
            "missing_png": sorted(required_figures - present_png),
            "missing_svg": sorted(required_figures - present_svg),
        },
        "realification_operator_norm_relative_error_max": {
            "passed": maximum_norm_error
            <= float(gate_spec["realification_operator_norm_relative_error_max"]),
            "value": maximum_norm_error,
            "threshold": gate_spec["realification_operator_norm_relative_error_max"],
        },
        "realification_singular_duplication_relative_error_max": {
            "passed": maximum_singular_error
            <= float(
                gate_spec["realification_singular_duplication_relative_error_max"]
            ),
            "value": maximum_singular_error,
            "threshold": gate_spec[
                "realification_singular_duplication_relative_error_max"
            ],
        },
        "minimum_png_count": {
            "passed": len(present_png) >= int(gate_spec["minimum_png_count"]),
            "value": len(present_png),
            "threshold": gate_spec["minimum_png_count"],
        },
        "minimum_svg_count": {
            "passed": len(present_svg) >= int(gate_spec["minimum_svg_count"]),
            "value": len(present_svg),
            "threshold": gate_spec["minimum_svg_count"],
        },
        "html_report_required": {
            "passed": True,
            "value": str(output_dir / "index.html"),
        },
    }
    return {
        "schema_version": 1,
        "specification": spec,
        "passed": all(bool(detail["passed"]) for detail in gates.values()),
        "gates": gates,
    }


def _file_manifest(output_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": "experiments.tran_vu",
        "entrypoint": "python -m experiments.tran_vu.run",
        "files": files,
    }


def run_characterization(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    sweep_points: int = 61,
    map_gap_points: int = 61,
    map_coupling_points: int = 51,
) -> dict[str, object]:
    """Run the complete characterization and return its acceptance report."""

    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    regime_rows = [_regime_row(fixture) for fixture in regime_fixtures()]
    directional_rows = _directional_coupling_sweep(sweep_points)
    gap_rows = _gap_sweep(sweep_points)
    perturbation_rows = _perturbation_sweep(sweep_points)
    halving_rows = _halving_rank_sweep()
    admission_rows, gap_ratios, coupling_ratios, admission_grid = _admission_map(
        map_gap_points,
        map_coupling_points,
    )
    nonnormality_rows = _nonnormality_sweep(max(17, sweep_points // 2))
    realification_rows = _realification_validation()

    datasets = {
        "regime_matrix": regime_rows,
        "directional_coupling_sweep": directional_rows,
        "gap_sweep": gap_rows,
        "perturbation_sweep": perturbation_rows,
        "halving_rank_sweep": halving_rows,
        "admission_map": admission_rows,
        "nonnormality_sweep": nonnormality_rows,
        "realification_validation": realification_rows,
    }
    for name, rows in datasets.items():
        _write_json(root / f"{name}.json", rows)
        _write_csv(root / f"{name}.csv", rows)

    _plot_regime_bounds(regime_rows, figure_dir)
    _plot_directional_coupling(directional_rows, figure_dir)
    _plot_gap_sweep(gap_rows, figure_dir)
    _plot_perturbation_sweep(perturbation_rows, figure_dir)
    _plot_halving_rank(halving_rows, figure_dir)
    _plot_admission_map(gap_ratios, coupling_ratios, admission_grid, figure_dir)
    _plot_observed_vs_bounds(regime_rows, figure_dir)
    _plot_nonnormality(nonnormality_rows, figure_dir)
    _plot_realification(realification_rows, figure_dir)

    figure_stems = tuple(path.stem for path in sorted(figure_dir.glob("*.png")))
    acceptance = _evaluate_acceptance(root, regime_rows, realification_rows)
    _write_html_report(root, regime_rows, acceptance, figure_stems)
    acceptance["gates"]["html_report_required"]["passed"] = (
        root / "index.html"
    ).exists()
    acceptance["passed"] = all(
        bool(detail["passed"]) for detail in acceptance["gates"].values()
    )
    _write_json(root / "acceptance_report.json", acceptance)

    summary = {
        "schema_version": 1,
        "acceptance_passed": acceptance["passed"],
        "regime_count": len(regime_rows),
        "theorem_applicable_count": sum(
            bool(row["theorem_applicable"]) for row in regime_rows
        ),
        "admitted_count": sum(bool(row["admitted"]) for row in regime_rows),
        "figure_count": len(figure_stems),
        "datasets": {name: len(rows) for name, rows in datasets.items()},
        "report": "index.html",
    }
    _write_json(root / "report.json", summary)
    _write_json(root / "manifest.json", _file_manifest(root))
    return acceptance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize Tran--Vu moderate-gap subspace certificates."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller deterministic sweeps while preserving every output class.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return a zero exit code even when an acceptance gate fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = (
        {"sweep_points": 13, "map_gap_points": 15, "map_coupling_points": 11}
        if args.quick
        else {}
    )
    acceptance = run_characterization(args.output, **settings)
    print(json.dumps(acceptance, indent=2, default=_json_default))
    print(f"[REPORT] {(args.output / 'index.html').resolve()}")
    if acceptance["passed"] or args.allow_failures:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
