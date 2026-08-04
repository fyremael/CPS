import numpy as np

from cps.pythia.analysis import analyze_reduced_operator
from cps.pythia.basis import BasisVector


def test_coupling_record_contains_subspace_stability_summary():
    matrix = np.array(
        [
            [1000.0, 0.5, 0.0],
            [0.0, 996.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    basis = tuple(
        BasisVector(
            name=f"basis-{index}",
            vector=None,
            parameter_name=f"parameter-{index}",
            component="theta",
        )
        for index in range(3)
    )

    records = analyze_reduced_operator(
        matrix,
        basis,
        phase_count=9,
        finite_horizon=2,
        compute_kreiss=False,
        maximum_couplings=1,
    )

    assert len(records) == 1
    report = records[0].subspace_stability
    assert report is not None
    assert report["theorem"] == "Tran--Vu (2025), Theorem 2.3"
    assert report["sample_count"] == 9
    assert report["skipped_near_reference_count"] >= 1
