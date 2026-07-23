import json
from pathlib import Path


def _planning_notebook() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "notebooks" / "05_cps_planning.ipynb").read_text())


def test_planning_notebook_does_not_assume_shared_colab_runtime():
    notebook = _planning_notebook()
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "acf0dbef")
    source = "".join(cell["source"])

    assert "CPS_EVIDENCE_PATH" in source
    assert "files.upload()" in source
    assert "cps-export*.zip" in source
    assert "shutil.unpack_archive" in source
    assert "/content/CPS/cps-artifacts" in source
    assert "/content/drive/MyDrive/cps-artifacts" in source
    assert "max(paths, key=packet_rank)" in source


def test_planning_notebook_exports_the_selected_packet_with_recommendation():
    notebook = _planning_notebook()
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "e6ff75bd")
    source = "".join(cell["source"])

    assert "export_artifacts(sources=(path.parent,))" in source
