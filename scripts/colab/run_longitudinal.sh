#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_notebook.sh" notebooks/02_pythia_longitudinal.ipynb "${1:-A100}" "${2:-cps-pythia-longitudinal}"
