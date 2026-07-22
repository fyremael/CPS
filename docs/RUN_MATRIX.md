# Run Matrix

## A. Synthetic exactness

| ID | Operator | Purpose | Dimensions | Sweep | Required result |
|---|---|---|---:|---|---|
| A1 | 2x2 analytic family | Verify eigenvalue formula and derivative | 2 | scalar phase | numerical/analytic agreement |
| A2 | triangular DAG | Verify cycle-support criterion | 4–32 | scalar phase | invariant spectrum |
| A3 | diamond DAG | Separate spectral motion from transient interference | 4–32 | scalar phase | fixed spectrum, varying gain |
| A4 | feedback cycle | Test unit-circle crossing | 4–32 | scalar phase | phase-dependent stability |
| A5 | Jordan-near operator | Test conditioning and branch tracking | 4–64 | scalar/block | condition blow-up recovered |
| A6 | random non-normal ensemble | Rank and grid convergence | 64–512 | top-k couplings | convergence envelope |

## B. Optimizer exact models

| ID | Objective | Optimizer | Variables | Purpose |
|---|---|---|---:|---|
| B1 | quadratic | GD | 16–256 | nominal sanity |
| B2 | quadratic | heavy-ball | 16–256 | augmented-state non-normality |
| B3 | quadratic | Nesterov | 16–256 | transient comparison |
| B4 | quadratic | diagonal adaptive | 16–256 | curvature/preconditioner misalignment |
| B5 | quadratic | block preconditioner | 64–512 | block partition recovery |

## C. Pythia instrument ladder

| ID | Subject | Checkpoints | Moment class | Rank | Phase samples | Hardware | Gate |
|---|---|---|---|---:|---:|---|---|
| P0.1 | Pythia-70M | step0, step1 | reconstructed | 3, 4, 8 | 9, 17 | CPU/T4 | smoke/replay |
| P0.2 | Pythia-70M | step1000 | reconstructed | 8, 16 | 17, 33 | T4 | autodiff/FD JVP |
| P0.3 | native Pythia-70M | step1000 | native | 8, 16 | 17, 33 | T4 | moment import |
| P1.1 | Pythia-70M | 0,1,2,4,...,512 | native where available | 16, 32 | 33 | T4/A100 | early longitudinal |
| P1.2 | Pythia-70M | every 1000 to 143000 | native | 32, 64 | 33 | A100 | dense longitudinal |
| P2.1 | PolyPythia-70M | common sparse grid, all seeds | native/reconstructed declared | 32, 64 | 33 | A100 | seed reproducibility |
| P2.2 | PolyPythia-70M | densified around events | same | 64 | 65 | A100 | outlier precursor |
| P3.1 | PolyPythia-160M controls | common sparse grid | same | 64 | 33 | A100 | seed causal decomposition |
| P4.1 | Pythia/PolyPythia-160M | sparse grid | same | 64, 128 | 33 | A100 | scale transfer |
| P4.2 | Pythia/PolyPythia-410M | sparse grid | same | 64, 128 | 33 | A100/H100 | scale transfer |
| P5.1 | Pythia-1B/2.8B | preregistered checkpoints | same | 128 | 33 | H100 | sparse external validation |

Checkpoint expansion is conditional. P0 must pass before P1; P1 before P2; prediction policy is frozen before P4.

## D. Basis and coordinate ablations

| Axis | Values |
|---|---|
| parameter selection | one matrix, attention projection, MLP projection, layer pair |
| state components | theta only; theta+m; theta+m+log-v |
| basis source | semantic axes; update direction; recent gradients; Krylov; random control |
| projection | orthogonal Galerkin; two-sided where supported |
| selected numel | 2k, 10k, 50k, 100k |
| rank | 3, 4, 8, 16, 32, 64, 128 |
| batch source | fixed built-in text; exact reconstructed stream; independent probe batches |
| JVP | autodiff; centered finite difference |
| precision | float32 reduced analysis; bf16/fp16 model with float32 diagnostic state |

## E. Spectroscopy ablations

- scalar entry versus singular-channel block families;
- complex phase continuation versus real paired rotation;
- phase counts 9, 17, 33, 65;
- horizons 5, 10, 20, 50;
- top-k coupling selection by magnitude versus semantic registry;
- with and without eigenpair tracking;
- with and without trace-moment features;
- frozen-step versus short-horizon monodromy;
- nominal operator versus phase envelope;
- random norm-matched structured perturbations.

## F. Prediction protocol

For each checkpoint, construct labels from a strictly future horizon. Split by seed/run, never by individual row. Compare:

1. conventional diagnostics only;
2. conventional diagnostics plus nominal operator features;
3. conventional diagnostics plus CPS phase-envelope features;
4. CPS-only diagnostic model as an interpretability control.

Report grouped ROC-AUC, average precision, calibration, lead time, confidence intervals, and failure cases. Thresholds and feature families are frozen before the held-out scale-transfer subjects.

## G. Matched continuation matrix

For selected checkpoints run identical-data forks:

1. baseline AdamW;
2. global learning-rate control;
3. block learning-rate control;
4. beta1/momentum control;
5. second-moment denominator floor;
6. clipping control;
7. damping/isotropization control when the optimizer representation supports it.

Change one mechanism per intervention experiment. Record loss, gradient norm, update norm, clipping/skipped-step events, wall time, tokens, and post-intervention CPS reserve.

## H. Colab execution matrix

| Notebook | Purpose | Default accelerator | Expected export |
|---|---|---|---|
| `00_instrument_smoke.ipynb` | core numerical fixture | CPU | smoke metrics |
| `01_pythia_70m_probe.ipynb` | one Pythia probe | T4 | evidence packet |
| `02_pythia_longitudinal.ipynb` | checkpoint sequence | A100 | longitudinal manifest |
| `03_polypythia_seed_study.ipynb` | replicated seeds | A100 | campaign feature table |
| `04_native_optimizer_state.ipynb` | download/reconstruct native moments | T4/CPU | native-state audit |
| `05_cps_planning.ipynb` | planner from prior packets | CPU | proposal ledger |
| `06_matched_continuation.ipynb` | causal continuation | T4/A100 | paired run records |

Every notebook must export `/content/cps-export`, and every CLI wrapper must save an execution log and stop the runtime.
