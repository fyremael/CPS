#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_notebook.sh" notebooks/03_polypythia_seed_study.ipynb "${1:-A100}" "${2:-cps-polypythia-seeds}"
