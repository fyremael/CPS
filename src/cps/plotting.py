from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .spectra import SweepResult


def plot_eigenvalue_trajectories(
    result: SweepResult,
    output: str | Path,
    unit_circle: bool = True,
    title: str | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for branch in result.eigenvalues.T:
        ax.plot(branch.real, branch.imag, linewidth=1.5)
        ax.scatter(branch[0].real, branch[0].imag, s=20)
    if unit_circle:
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=0.9)
    ax.axhline(0.0, linewidth=0.6)
    ax.axvline(0.0, linewidth=0.6)
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or f"CPS trajectory for coupling ({result.row}, {result.col})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_heatmap(
    values: ArrayLike,
    output: str | Path,
    title: str,
    label: str,
) -> Path:
    arr = np.asarray(values, dtype=float)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(arr)
    ax.set_xlabel("Source coordinate")
    ax.set_ylabel("Target coordinate")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
