"""Coupling-Phase Spectroscopy (CPS).

CPS probes a reduced dynamical operator by rotating selected couplings while
preserving their magnitude, then measures spectral motion, transient growth,
and robustness margins.
"""

from .metrics import (
    CPSMetrics,
    compute_cps_metrics,
    eigenvalue_condition_numbers,
    finite_horizon_gain,
    kreiss_surrogate,
    minimum_eigenvalue_gap,
)
from .moments import phase_moment_spectrum, trace_moment
from .perturbations import (
    phase_rotate_entry,
    phase_sweep_entry,
    real_pair_rotation,
    rotate_block_svd_phase,
)
from .projection import arnoldi_projection, project_dense_operator, randomized_projection
from .spectra import SweepResult, spectral_sweep

__all__ = [
    "CPSMetrics",
    "SweepResult",
    "arnoldi_projection",
    "compute_cps_metrics",
    "eigenvalue_condition_numbers",
    "finite_horizon_gain",
    "kreiss_surrogate",
    "minimum_eigenvalue_gap",
    "phase_moment_spectrum",
    "phase_rotate_entry",
    "phase_sweep_entry",
    "project_dense_operator",
    "randomized_projection",
    "real_pair_rotation",
    "rotate_block_svd_phase",
    "spectral_sweep",
    "trace_moment",
]
