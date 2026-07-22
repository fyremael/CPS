import json
from pathlib import Path


def test_seed_study_uses_schema_v2_run_key_and_safe_metric_maxima():
    notebook = json.loads(
        Path("notebooks/03_polypythia_seed_study.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert 'manifest["run_spec"]["name"]' not in source
    assert 'run_spec.get("key")' in source
    assert 'run_spec.get("name")' in source
    assert 'default=float("nan")' in source
    assert 'manifest.get("jacobian", {})' in source
