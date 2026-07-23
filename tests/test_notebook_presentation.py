from cps import notebook


def test_release_theme_renders_css(monkeypatch):
    captured = []
    monkeypatch.setattr(notebook, "_display_html", captured.append)
    monkeypatch.setattr(notebook, "_THEME_APPLIED", False)

    notebook.apply_release_theme()
    notebook.apply_release_theme()

    assert len(captured) == 1
    assert ".cps-stage" in captured[0]


def test_stage_banner_escapes_content(monkeypatch):
    captured = []
    monkeypatch.setattr(notebook, "_display_html", captured.append)
    monkeypatch.setattr(notebook, "_THEME_APPLIED", False)

    notebook.stage_banner(
        "1",
        "Measure <operator>",
        objective="Preserve A & B",
        deliverable="packet",
    )

    rendered = captured[-1]
    assert "Measure &lt;operator&gt;" in rendered
    assert "A &amp; B" in rendered
    assert "Evidence produced" in rendered
