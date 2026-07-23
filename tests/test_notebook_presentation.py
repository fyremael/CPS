from cps import notebook


def test_release_theme_renders_css(monkeypatch):
    captured = []
    monkeypatch.setattr(notebook, "_display_html", captured.append)
    notebook.apply_release_theme()
    assert captured
    assert ".cps-stage" in captured[0]


def test_stage_banner_escapes_content(monkeypatch):
    captured = []
    monkeypatch.setattr(notebook, "_display_html", captured.append)
    notebook.stage_banner(
        "1",
        "Measure <operator>",
        objective="Preserve A & B",
        deliverable="packet",
    )
    rendered = captured[0]
    assert "Measure &lt;operator&gt;" in rendered
    assert "A &amp; B" in rendered
    assert "Evidence produced" in rendered
