# Colab CLI execution contract

CPS uses the official `google-colab-cli` for accelerator provisioning and notebook execution.

## Installation

```bash
uv tool install google-colab-cli
# or
pip install google-colab-cli
```

The CLI currently targets Linux and macOS. Authenticate once according to the upstream instructions.

## Lifecycle

The wrapper in `scripts/colab/run_notebook.sh` performs:

1. `colab new -s NAME --gpu GPU`;
2. `colab exec -s NAME -f NOTEBOOK`;
3. `colab log` to `.ipynb` and `.jsonl`;
4. `colab download` of `/content/cps-export`;
5. `colab stop`, including failure paths through a shell trap.

The notebook itself clones the repository at `CPS_GIT_REF`, installs the Pythia extra, runs the governed configuration, and places exportable results under `/content/cps-export`.

## Environment controls

- `CPS_REPO_URL`: repository URL, default `https://github.com/fyremael/CPS.git`.
- `CPS_GIT_REF`: branch or tag, default `main`.
- `CPS_REVISIONS`: comma-separated checkpoint list for longitudinal runs.
- `CPS_SEEDS`: comma-separated PolyPythia seeds.
- `CPS_NATIVE_REVISION`: raw optimizer checkpoint revision.
- `CPS_CONTINUATION_STEPS`: matched continuation horizon.
- `CPS_LR_SCALE`, `CPS_BETA1`: intervention values.

## Artifact rule

The local execution log is part of the evidence packet. A numerical result without the notebook log, repository commit, configuration, and manifest is non-admissible.
