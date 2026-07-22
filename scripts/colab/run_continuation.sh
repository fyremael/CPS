#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_notebook.sh" notebooks/06_matched_continuation.ipynb "${1:-T4}" "${2:-cps-pythia-continuation}"
