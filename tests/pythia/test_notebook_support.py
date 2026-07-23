import json
from pathlib import Path

import numpy as np

from cps.pythia.notebook_support import (
    build_probe_for_notebook,
    load_evidence_packet,
    select_candidate_edges,
    write_planner_recommendation,
)
from cps.pythia.planner import PlannerRecommendation


def test_release_probe_contract_is_compact_and_self_contained():
    config = build_probe_for_notebook(revision="step0", run_name="release-test")
    assert config.model.revision == "step0"
    assert config.jacobian.mode == "finite_difference"
    assert config.output.run_name == "release-test"


def test_load_packet_and_prefer_declared_couplings(tmp_path: Path):
    root = tmp_path / "packet"
    root.mkdir()
    matrix = np.array([[1.0, 0.7], [0.2, 0.9]])
    np.save(root / "reduced_operator.npy", matrix)
    (root / "couplings.json").write_text(
        json.dumps([{"row": 1, "col": 0}, {"row": 0, "col": 1}]),
        encoding="utf-8",
    )
    packet = load_evidence_packet(root)
    assert np.allclose(packet.matrix, matrix)
    assert select_candidate_edges(packet, maximum=2) == [(1, 0), (0, 1)]


def test_write_planner_recommendation(tmp_path: Path):
    recommendation = PlannerRecommendation(
        control="learning_rate_scale",
        baseline=1.0,
        recommended=0.9,
        baseline_risk=1.2,
        recommended_risk=1.0,
        rationale="test",
    )
    target = write_planner_recommendation(
        tmp_path,
        recommendation,
        selected_edges=[(0, 1)],
        candidate_grid=[1.0, 0.9],
        control_operationalization={"translation": "gamma = 1 - scale"},
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["recommended"] == 0.9
    assert payload["selected_edges"] == [[0, 1]]
