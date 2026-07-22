import json

from cps.pythia.features import extract_checkpoint_features


def test_extract_checkpoint_features(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_spec": {"key": "pythia-70m", "seed": None},
                "revision": "step1000",
                "selected_parameter_numel": 10,
                "basis_rank": 2,
                "projection": {"maximum_closure_residual": 0.2},
            }
        )
    )
    metric = {
        "spectral_radius_max": 1.1,
        "finite_horizon_gain": 2.0,
        "minimum_gap": 0.3,
        "maximum_eigenvalue_condition": 4.0,
        "total_loop_length": 0.5,
        "baseline_displacement_max": 0.4,
    }
    (tmp_path / "couplings.json").write_text(
        json.dumps([{"metrics": metric}, {"metrics": {**metric, "spectral_radius_max": 1.2}}])
    )
    features = extract_checkpoint_features(tmp_path)
    assert features.step == 1000
    assert features.cps_spectral_radius_max == 1.2
    assert features.coupling_count == 2
