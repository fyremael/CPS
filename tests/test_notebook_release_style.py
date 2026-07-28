from pathlib import Path

import nbformat


def notebook_text(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_all_notebooks_are_release_grade_and_export_consistently():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(paths) == 8
    for path in paths:
        text = notebook_text(path)
        assert "Grand Challenge Labs · Coupling-Phase Spectroscopy" in text
        assert "## Release contract" in text
        assert "## Interpretation checklist" in text
        assert "stage_banner(" in text
        assert "apply_release_theme()" in text
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


def test_tran_vu_characterization_is_self_contained_and_visual():
    characterization = notebook_text(Path("notebooks/07_tran_vu_characterization.ipynb"))
    assert "run_characterization" in characterization
    assert "all nine plots" in characterization
    assert "figures" in characterization
    assert "index.html" in characterization
    assert "acceptance[\"passed\"]" in characterization


def test_release_titles_are_numbered_and_concise():
    for index, path in enumerate(sorted(Path("notebooks").glob("*.ipynb"))):
        notebook = nbformat.read(path, as_version=4)
        first_line = notebook.cells[0].source.splitlines()[0]
        assert first_line.startswith(f"# {index:02d} · ")
        assert len(first_line) <= 48


def test_release_style_document_is_present():
    text = Path("docs/NOTEBOOK_RELEASE_STYLE.md").read_text(encoding="utf-8")
    assert "Language doctrine" in text
    assert "Visual doctrine" in text
    assert "Self-containment contract" in text
    assert "/content/cps-export.zip" in text
