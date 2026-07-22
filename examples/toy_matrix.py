from pathlib import Path

import numpy as np

from cps.plotting import plot_eigenvalue_trajectories
from cps.spectra import spectral_sweep


def main() -> None:
    # A stable but non-normal reduced update operator.
    a = np.array(
        [
            [0.72, 0.90, 0.00, 0.00],
            [-0.15, 0.68, 0.65, 0.00],
            [0.00, -0.30, 0.76, 0.80],
            [0.22, 0.00, -0.18, 0.64],
        ],
        dtype=float,
    )
    result = spectral_sweep(a, 0, 1, compute_kreiss=True)
    output = Path("artifacts/toy_trajectory.png")
    plot_eigenvalue_trajectories(result, output, title="Coupling-phase trajectory of a reduced update operator")
    print(result.metrics.to_dict())
    print(output)


if __name__ == "__main__":
    main()
