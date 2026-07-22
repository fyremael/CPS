#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_notebook.sh" notebooks/01_pythia_70m_probe.ipynb "${1:-T4}" "${2:-cps-pythia-70m-smoke}"
