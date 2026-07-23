from pathlib import Path

import nbformat


def notebook_text(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_all_notebooks_are_release_grade_and_export_consistently():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(paths) == 7
    for path in paths:
        text = notebook_text(path)
        assert "Grand Challenge release notebook" in text
        assert "## Release contract" in text
        assert "## Interpretation checklist" in text
        assert "stage_banner(" in text
        assert "archive = export_artifacts()" in text
        assert "/content/cps-export.zip" in text


def test_planning_and_continuation_are_self_contained():
    planning = notebook_text(Path("notebooks/05_cps_planning.ipynb"))
    continuation = notebook_text(Path("notebooks/06_matched_continuation.ipynb"))
    assert "run_self_contained_probe" in planning
    assert "CPS_EVIDENCE_PATH" in planning
    assert "run_self_contained_probe" in continuation
    assert "plan_scalar_control" in continuation
    assert "recommendation.recommended" in continuation
