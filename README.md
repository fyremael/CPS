# Coupling-Phase Spectroscopy

**Coupling-Phase Spectroscopy (CPS)** is a structured pseudospectral diagnostic for projected optimizer-state Jacobians. It asks one controlled question:

> Holding a coupling's strength fixed, how much can its orientation change the local modes, transient amplification, and stability reserve of training?

The repository contains the mathematical instrument, a functional PyTorch optimizer-state implementation, a Pythia/PolyPythia subject adapter, Colab runner notebooks, campaign aggregation and prediction tools, matched continuation harnesses, tests, and the preprint source.

## Research object

Freeze a checkpoint, minibatch, and algorithmic randomness. Let the complete optimizer state be

\[
z_t=(\theta_t,m_t,v_t,\ldots),
\qquad z_{t+1}=F_t(z_t),
\qquad J_t=D F_t(z_t).
\]

CPS never materializes the full Jacobian. It selects dimensionless optimizer coordinates, constructs semantic probe directions, obtains Jacobian-vector products, and forms

\[
\widehat J_t=Q_t^*J_tQ_t.
\]

For a reduced coupling or singular channel, CPS preserves its magnitude and rotates its phase. It then records spectral migration, finite-horizon gain, minimum mode separation, eigenvalue conditioning, loop geometry, and projection closure residuals.

## Pythia programme

The primary empirical ladder is:

1. **Pythia-70M** — instrument validation and dense longitudinal analysis.
2. **PolyPythia-70M** — seed replication and prospective outlier prediction.
3. **PolyPythia-160M causal controls** — initialization versus data-order attribution.
4. **Pythia-160M/410M** — scale-transfer gate.
5. **Pythia-1B/2.8B** — sparse validation after the smaller gates pass.

Pythia weights are loaded from Hugging Face revisions. Native GPT-NeoX checkpoints can be downloaded and their ZeRO-partitioned Adam moments reconstructed offline. Every run declares whether optimizer moments are **native** or **reconstructed**; those evidence classes must not be conflated.

## Installation

Core matrix instrument:

```bash
python -m pip install -e .
```

Pythia runners:

```bash
python -m pip install -e '.[pythia,notebooks]'
```

Development:

```bash
python -m pip install -e '.[pythia,notebooks,dev]'
pytest -q
python scripts/validate_notebooks.py
```

## Local and Colab execution

Run a local probe:

```bash
cps-pythia probe subjects/pythia/configs/pythia_70m_smoke.yaml
```

Run a checkpoint sequence:

```bash
cps-pythia longitudinal \
  subjects/pythia/configs/pythia_70m_longitudinal.yaml \
  --revisions step0 step1 step16 step512 step1000
```

The official Google Colab CLI can provision accelerators and execute notebooks directly from a terminal. After installing and authenticating it:

```bash
./scripts/colab/run_pythia_70m_smoke.sh T4
./scripts/colab/run_longitudinal.sh A100
./scripts/colab/run_seed_study.sh A100
./scripts/colab/run_continuation.sh T4
```

The wrapper always exports the Colab execution log, retrieves `/content/cps-export`, and tears down the runtime.


## Notebook pedagogy and live execution

> **Colab import bootstrap.** The notebooks add `/content/CPS/src` to the live kernel path after installation. This is required because a newly created editable-install `.pth` file is not reprocessed automatically by an already-running kernel. No runtime restart is required. The bootstrap prints the resolved `cps.__file__`; subsequent cells should be run only after that confirmation appears.


The seven Colab notebooks are written as executable lessons rather than opaque launchers. They show the resolved configuration, runtime inventory, evidence boundary, active attention and JVP backends, per-column projection progress, coupling-sweep progress, result tables, diagnostic figures, and artifact locations. The same line-oriented progress stream is preserved by `colab-cli log`, making long remote runs auditable while they execute.

The Pythia-70M notebook also explains the fused-SDPA forward-AD limitation and displays whether the run used exact `torch.func.jvp` or the declared centered finite-difference fallback.

The editorial, visual, self-containment, and export requirements are specified in [`docs/NOTEBOOK_RELEASE_STYLE.md`](docs/NOTEBOOK_RELEASE_STYLE.md).

## Native optimizer states

The Transformers-compatible Pythia repositories and the native GPT-NeoX optimizer-state repositories do not necessarily expose the same Hub refs. A semantic request for the final checkpoint remains `step143000`, but a current native repository may store those files only on `main`.

Download the raw files required for moment reconstruction:

```bash
cps-pythia download-native \
  EleutherAI/neox-ckpt-pythia-70m \
  step143000 \
  /data/native-pythia-70m-step143000
```

The downloader lists the repository refs before downloading. It prefers a real `step143000` ref when present; otherwise, for a Pythia native repository exposing only `main`, it resolves the final-step request to `main` and records both revisions in `cps_download_manifest.json`. Missing intermediate native revisions fail closed rather than substituting a weight-only checkpoint.

Inspect the native checkpoint:

```bash
cps-pythia inspect-native /data/native-pythia-70m-step143000
```

Then use `moment_source: native` and set `native_checkpoint_dir` in the run configuration. Native state files are pickle-based external artifacts; load only trusted repositories.

## Campaign outputs

Each probe writes an immutable evidence packet:

```text
manifest.json          configuration, environment, checkpoint and evidence class
reduced_operator.npy   projected dimensionless optimizer-state Jacobian
basis.json             semantic direction registry
couplings.json         ranked CPS measurements
```

Aggregate a campaign:

```bash
cps-pythia aggregate artifacts/pythia artifacts/pythia/features.csv
```

Evaluate incremental predictive value after joining future-event labels and conventional diagnostics:

```bash
cps-pythia evaluate artifacts/pythia/labeled_features.csv \
  --label-column future_spike \
  --group-column seed \
  --output-json artifacts/pythia/prediction_report.json
```

## Matched intervention test

A planning recommendation is not accepted as evidence until it survives a matched continuation:

```bash
cps-pythia continuation \
  --model-id EleutherAI/pythia-70m \
  --revision step1000 \
  --steps 20 \
  --lr-scale 0.8
```

The baseline and intervention forks begin from identical weights and consume identical subsequent batches. The current continuation harness supports learning-rate, momentum, denominator, and clipping controls. Exact historical continuation additionally requires complete native optimizer-state import and the original token stream.

## Repository map

```text
src/cps/                 core CPS instrument
src/cps/pythia/          functional optimizer map and Pythia subject adapter
subjects/pythia/         governed registries and experiment configurations
notebooks/               Colab-executable runner harnesses
scripts/colab/           official Colab CLI lifecycle wrappers
experiments/             synthetic validation and figure generation
manuscript/              preprint source and reproducible build
scripts/                 notebook and repository validation
tests/                    core, optimizer-map, native-state and subject tests
docs/                     programme contracts, metrics, run matrix and ADRs
```

## Evidence discipline

CPS is falsifiable. It must add held-out predictive value beyond gradient norms, loss trends, Hessian or generalized curvature estimates, nominal spectral radius, and nominal finite-horizon gain. A sophisticated phase sweep that does not improve prediction or intervention outcomes remains a research visualization, not an optimizer controller.

## Safety and resource controls

- Native checkpoints can approach gigabytes even at 70M scale; use explicit caches and quotas.
- Autodiff JVPs differentiate through the gradient and can be memory-intensive. Start with one matrix block and rank 3–4.
- Pythia probes request `attention_implementation: eager` because PyTorch fused efficient-SDPA kernels may not implement forward AD.
- `jacobian.autodiff_backend: auto` performs a preflight and records a centered finite-difference fallback rather than failing midway through projection.
- Finite-difference JVPs are slower and approximate; the effective backend and fallback reason are written to `manifest.json`.
- Do not commit downloaded model or optimizer checkpoints.
- Every Colab campaign must stop its runtime after artifact retrieval.

## Citation

See `CITATION.cff`. Build the manuscript PDF with `make manuscript`.
