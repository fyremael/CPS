PYTHON ?= python

.PHONY: test validate synthetic figures manuscript package

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_notebooks.py
	$(PYTHON) -m compileall -q src scripts tests

synthetic:
	PYTHONPATH=src $(PYTHON) experiments/synthetic_quadratics.py

figures:
	PYTHONPATH=src $(PYTHON) experiments/make_manuscript_figures.py

manuscript: figures
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error cps_preprint.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error cps_preprint.tex

package: test validate manuscript
	$(PYTHON) scripts/make_release.py
