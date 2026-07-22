"""Synthetic validation experiments for Coupling-Phase Spectroscopy.

Experiment A compares a normal contraction with an isospectral non-normal
contraction.  Their eigenvalues agree, but the non-normal operator has a much
larger finite-horizon gain.  The selected edge lies on no directed cycle, so its
phase sweep leaves the eigenvalues fixed while changing path interference.

Experiment B closes a feedback cycle.  The same phase sweep now moves the
spectrum and can expose a phase-dependent stability boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cps.metrics import finite_horizon_gain
from cps.perturbations import phase_sweep_entry
from cps.plotting import plot_eigenvalue_trajectories
from cps.spectra import spectral_sweep


def _plot_gain_curve(matrix: np.ndarray, edge: tuple[int, int], output: Path, title: str) -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, 129)
    matrices = phase_sweep_entry(matrix, *edge, phases)
    gains = [finite_horizon_gain(a, horizon=12) for a in matrices]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(phases / np.pi, gains, linewidth=1.6)
    ax.set_xlabel(r"Phase offset $\phi/\pi$")
    ax.set_ylabel("Finite-horizon gain")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(output_dir: Path) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    normal = np.diag([0.82, 0.74, 0.66, 0.58]).astype(float)
    # A directed acyclic "diamond": two paths interfere at the fourth coordinate.
    nonnormal_dag = normal.copy()
    nonnormal_dag[0, 1] = 1.10
    nonnormal_dag[0, 2] = 0.90
    nonnormal_dag[1, 3] = 1.00
    nonnormal_dag[2, 3] = 0.95

    # Add one feedback edge to turn path interference into spectral motion.
    cycle_coupled = nonnormal_dag.copy()
    cycle_coupled[3, 0] = 0.11

    cases = [
        ("normal", normal, (0, 1)),
        ("isospectral_nonnormal", nonnormal_dag, (0, 1)),
        ("cycle_coupled", cycle_coupled, (0, 1)),
    ]
    results: dict[str, dict[str, float]] = {}
    for name, matrix, edge in cases:
        result = spectral_sweep(matrix, *edge, compute_kreiss=True)
        results[name] = result.metrics.to_dict()
        plot_eigenvalue_trajectories(
            result,
            output_dir / f"{name}_trajectory.png",
            title=f"{name.replace('_', ' ').title()}: eigenvalue trajectories",
        )
        _plot_gain_curve(
            matrix,
            edge,
            output_dir / f"{name}_gain.png",
            f"{name.replace('_', ' ').title()}: transient gain under phase sweep",
        )

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(Path("artifacts/synthetic_quadratics")), indent=2))
