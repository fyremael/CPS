from __future__ import annotations

import json
from pathlib import Path

from cps.subspace_stability import leading_singular_subspace_stability
from experiments.tran_vu.fixtures import regime_fixtures
from experiments.tran_vu.run import _expectation_errors, run_characterization


def test_governed_regime_matrix_matches_declared_outcomes():
    fixtures = regime_fixtures()
    assert len(fixtures) == 10
    assert len({fixture.name for fixture in fixtures}) == len(fixtures)

    for fixture in fixtures:
        certificate = leading_singular_subspace_stability(
            fixture.reference,
            fixture.perturbed,
            rank=fixture.rank,
        )
        assert _expectation_errors(fixture, certificate) == ()


def test_characterization_package_emits_complete_visual_packet(tmp_path: Path):
    output = tmp_path / "tran-vu"
    acceptance = run_characterization(
        output,
        sweep_points=9,
        map_gap_points=11,
        map_coupling_points=9,
    )

    assert acceptance["passed"]
    assert (output / "index.html").is_file()
    assert (output / "acceptance_report.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "regime_matrix.csv").is_file()
    assert (output / "admission_map.json").is_file()

    png = sorted((output / "figures").glob("*.png"))
    svg = sorted((output / "figures").glob("*.svg"))
    assert len(png) == 9
    assert len(svg) == 9
    assert all(path.stat().st_size > 10_000 for path in png)
    assert all(path.stat().st_size > 1_000 for path in svg)

    report = (output / "index.html").read_text(encoding="utf-8")
    assert "Tran--Vu moderate-gap characterization" in report
    assert "Visual characterization" in report
    assert "Admission map" in report

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifested = {entry["path"] for entry in manifest["files"]}
    assert "index.html" in manifested
    assert "figures/06_admission_map.png" in manifested
    assert "figures/09_realification_validation.svg" in manifested


def test_acceptance_report_preserves_fail_closed_contract(tmp_path: Path):
    output = tmp_path / "tran-vu"
    run_characterization(
        output,
        sweep_points=7,
        map_gap_points=9,
        map_coupling_points=7,
    )
    report = json.loads((output / "acceptance_report.json").read_text(encoding="utf-8"))

    assert report["passed"]
    assert all(detail["passed"] for detail in report["gates"].values())

    regimes = json.loads((output / "regime_matrix.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in regimes}
    assert by_name["weak_directional_coupling"]["admitted"]
    assert not by_name["gap_failure"]["theorem_applicable"]
    assert "gap_below_four_perturbation_norms" in by_name["gap_failure"]["reasons"]
    assert not by_name["strong_local_coupling"]["admitted"]
    assert by_name["complex_realification"]["representation"] == "complex_realification"
