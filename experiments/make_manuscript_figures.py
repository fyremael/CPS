from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cps.metrics import finite_horizon_gain
from cps.perturbations import phase_sweep_entry
from cps.plotting import plot_eigenvalue_trajectories
from cps.spectra import spectral_sweep


def matrices():
    normal = np.diag([0.82, 0.74, 0.66, 0.58]).astype(float)
    dag = normal.copy()
    dag[0, 1] = 1.10
    dag[0, 2] = 0.90
    dag[1, 3] = 1.00
    dag[2, 3] = 0.95
    cycle = dag.copy()
    cycle[3, 0] = 0.11
    return normal, dag, cycle


def pipeline(output: Path):
    labels = [
        "Checkpoint and\noptimizer state",
        "Frozen optimizer map\nand JVPs",
        "Projected Jacobian\n$\\widehat J=W^*JV$",
        "Magnitude-preserving\nphase families",
        "Spectral and\ntransient observables",
        "Damping, momentum,\npreconditioner, blocks",
    ]
    fig, ax = plt.subplots(figsize=(11.2, 2.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    xs = np.linspace(0.2, 10.2, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        box = FancyBboxPatch(
            (x, 0.75),
            1.55,
            1.0,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.0,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 0.775, 1.25, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + 1.57, 1.25),
                (xs[i + 1] - 0.03, 1.25),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
            )
            ax.add_patch(arrow)
    ax.text(6.0, 2.18, "Coupling-Phase Spectroscopy", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def gain_comparison(output: Path):
    normal, dag, cycle = matrices()
    phases = np.linspace(0.0, 2.0 * np.pi, 129)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for label, matrix in [("normal", normal), ("isospectral non-normal", dag), ("cycle-coupled", cycle)]:
        mats = phase_sweep_entry(matrix, 0, 1, phases)
        gains = [finite_horizon_gain(a, horizon=12) for a in mats]
        ax.plot(phases / np.pi, gains, label=label, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Phase offset $\phi/\pi$")
    ax.set_ylabel(r"$\max_{1\leq k\leq 12}\|J(\phi)^k\|_2$")
    ax.set_title("Equal nominal spectral radius does not imply equal dynamical reserve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def heatmap(output: Path):
    _, _, a = matrices()
    n = a.shape[0]
    values = np.full((n, n), np.nan)
    phases = np.linspace(0.0, 2.0 * np.pi, 33)
    for i in range(n):
        for j in range(n):
            if i == j or abs(a[i, j]) < 1e-12:
                continue
            result = spectral_sweep(
                a, i, j, phases=phases, finite_horizon=12, compute_kreiss=False
            )
            m = result.metrics
            values[i, j] = m.spectral_radius_max + 0.1 * np.log1p(m.finite_horizon_gain)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(values)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("source coordinate $j$")
    ax.set_ylabel("target coordinate $i$")
    ax.set_title("Coupling risk map")
    fig.colorbar(image, ax=ax, label="composite CPS risk")
    for i in range(n):
        for j in range(n):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    output = Path("manuscript/figures")
    output.mkdir(parents=True, exist_ok=True)
    normal, dag, cycle = matrices()
    pipeline(output / "pipeline.png")
    gain_comparison(output / "gain_comparison.png")
    heatmap(output / "risk_heatmap.png")
    result = spectral_sweep(cycle, 0, 1, finite_horizon=12, compute_kreiss=True)
    plot_eigenvalue_trajectories(
        result,
        output / "cycle_trajectory.png",
        title="Phase-induced migration of optimizer modes",
    )


if __name__ == "__main__":
    main()
