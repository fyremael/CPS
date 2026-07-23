import json
from pathlib import Path


def _planning_notebook() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "notebooks" / "05_cps_planning.ipynb").read_text())


def _notebook_source() -> str:
    notebook = _planning_notebook()
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_planning_notebook_is_self_contained_by_default():
    source = _notebook_source()

    assert "run_self_contained_probe" in source
    assert "CPS_EVIDENCE_PATH" in source
    assert "No external evidence" not in source
    assert "Run a probe notebook first" not in source


def test_planning_notebook_preserves_optional_cross_runtime_reuse():
    source = _notebook_source()

    assert "load_evidence_packet" in source
    assert "ZIP archive" in source
    assert "Reusing explicit evidence" in source


def test_planning_notebook_exports_measurement_and_recommendation():
    source = _notebook_source()

    assert "planner_recommendation.json" in source
    assert "archive = export_artifacts()" in source
    assert "/content/cps-export.zip" in source
