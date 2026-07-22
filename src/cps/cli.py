from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .plotting import plot_eigenvalue_trajectories
from .spectra import spectral_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Coupling-Phase Spectroscopy on a .npy matrix")
    parser.add_argument("matrix", type=Path, help="Path to square NumPy .npy matrix")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--col", type=int, required=True)
    parser.add_argument("--phase-count", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--no-kreiss", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("cps_output"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    matrix = np.load(args.matrix)
    phases = np.linspace(0.0, 2.0 * np.pi, args.phase_count)
    result = spectral_sweep(
        matrix,
        args.row,
        args.col,
        phases=phases,
        finite_horizon=args.horizon,
        compute_kreiss=not args.no_kreiss,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics.to_dict(), indent=2), encoding="utf-8")
    np.save(args.output_dir / "eigenvalues.npy", result.eigenvalues)
    plot_eigenvalue_trajectories(result, args.output_dir / "trajectories.png")
    print(metrics_path)


if __name__ == "__main__":
    main()
