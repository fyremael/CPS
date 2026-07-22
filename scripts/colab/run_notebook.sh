#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK=${1:?usage: run_notebook.sh NOTEBOOK [GPU] [SESSION]}
GPU=${2:-T4}
SESSION=${3:-cps-$(basename "${NOTEBOOK}" .ipynb)}
ARTIFACT_DIR=${CPS_COLAB_ARTIFACT_DIR:-artifacts/colab/${SESSION}}
mkdir -p "${ARTIFACT_DIR}"

cleanup() {
  colab stop -s "${SESSION}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

colab new -s "${SESSION}" --gpu "${GPU}"
colab exec -s "${SESSION}" -f "${NOTEBOOK}"
colab log -s "${SESSION}" -o "${ARTIFACT_DIR}/execution.ipynb"
colab log -s "${SESSION}" -o "${ARTIFACT_DIR}/execution.jsonl"

# Notebooks place distributable outputs under /content/cps-export.
if colab ls -s "${SESSION}" /content/cps-export >/dev/null 2>&1; then
  colab download -s "${SESSION}" /content/cps-export "${ARTIFACT_DIR}/cps-export"
fi
