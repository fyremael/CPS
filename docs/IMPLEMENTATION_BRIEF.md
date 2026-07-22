# Implementation Brief

## 1. Architectural boundary

CPS analyzes a frozen one-step optimizer map. The production subject adapter must make the following inputs explicit:

```python
next_state, telemetry = optimizer_step(
    state=state,
    model_context=frozen_parameters,
    batch=batch,
    rng_state=rng_state,
    hyperparameters=hyperparameters,
)
```

The Pythia implementation currently differentiates a governed selected-coordinate AdamW map. Parameters outside the selection remain frozen model context. This gives an exact Jacobian for the selected map, while projection closure residuals expose leakage from the chosen reduced basis.

## 2. Package layout

```text
src/cps/
  perturbations.py       scalar and real phase families
  families.py            generic and singular-channel sweeps
  tracking.py            eigenvalue continuation
  metrics.py             spectral/transient observables
  moments.py             trace-moment spectroscopy
  projection.py          dense/random/Arnoldi reduction
  pythia/
    registry.py          subject and checkpoint registry
    config.py            typed YAML configuration
    blocks.py            semantic parameter registry
    state_layout.py      theta/m/log-v flattening
    functional_adamw.py  differentiable selected-coordinate update
    native_state.py      native ZeRO Adam-state reconstruction
    checkpoints.py       governed artifact download
    data.py              deterministic probe batches
    basis.py             semantic probe basis
    reduced_operator.py  matrix-free projection and residuals
    analysis.py          coupling selection and CPS records
    planner.py           intervention proposal contracts
    continuation.py      matched baseline/intervention forks
    features.py          checkpoint feature extraction
    prediction.py        grouped incremental-value evaluation
    variance.py          seed/step variance decomposition
    campaign.py          campaign aggregation and reports
    runner.py            probe and longitudinal orchestration
    cli.py               command-line surface
```

## 3. Optimizer-state coordinates

Use

\[
z=(\theta,m,u),\qquad u=\log(v+\tau).
\]

`StateLayout` is the sole authority for slices, shapes, flattening, and unflattening. No component may reconstruct offsets independently.

A positive diagonal scaling `S` defines dimensionless coordinates. JVPs are taken through

\[
\delta\widetilde F(x)=S\left[F(z+S^{-1}x)-F(z)\right],
\]

so the reduced operator represents \(SJS^{-1}\).

## 4. Functional AdamW contract

`FunctionalAdamWProbe` must:

- use `torch.func.functional_call` rather than mutate model parameters;
- compute gradients with respect to selected coordinates only;
- preserve an explicit frozen batch;
- encode native or reconstructed first and second moments;
- perform bias correction at the declared optimizer step;
- return a differentiable next state;
- expose autodiff and finite-difference JVP paths.

Native-state import is evidence-sensitive. Pickle-based checkpoints are loaded only through an explicit trusted-artifact path, and manifests declare source files and partition count.

## 5. Projection contract

Let `basis = [q_1, ..., q_r]` be orthonormal dimensionless directions. The reduced operator is

\[
A_{ij}=q_i^*Jq_j.
\]

For each column, persist

\[
r_j=Jq_j-QQ^*Jq_j,
\]

and report absolute and relative closure residuals. A low-rank result without these residuals is inadmissible evidence.

Basis sources include:

- parameter, first-moment, and second-moment semantic block directions;
- current update direction;
- recent gradients or updates when supplied by a campaign;
- Krylov directions;
- random controls.

## 6. Phase-family contract

Implement and identify separately:

1. scalar complex phase;
2. real paired-coupling rotation;
3. singular-channel block phase;
4. block basis-misalignment rotation.

A complex continuation is a measurement device. Only real optimizer controls may be credited as interventions.

## 7. Spectral engine contract

The engine must:

- never sort branches independently by real part or modulus;
- use overlap/distance assignment for continuation;
- verify unordered spectral closure at a complete turn;
- record branch permutations;
- use double precision for reduced analysis;
- refine around radius maxima, gap minima, or conditioning events when configured;
- report spectral radius, finite-horizon gain, gaps, conditioning, loop geometry, numerical radius, departure from normality, Kreiss surrogate, and trace harmonics.

## 8. Pythia runner contract

One probe performs:

1. load model/tokenizer revision;
2. select semantic parameters and enforce a numel budget;
3. construct deterministic batch;
4. load native moments or declare reconstruction;
5. build dimensionless state and semantic basis;
6. obtain matrix-free JVP columns;
7. form reduced operator and closure report;
8. select governed couplings;
9. execute phase sweeps;
10. write an immutable evidence packet.

The runner must fail closed when a parameter pattern matches nothing, selected numel exceeds budget, native moments cannot be aligned, or the JVP mode is unsupported.

## 9. Evidence packet

```text
artifacts/pythia/<campaign>/<run>/<revision>/
  manifest.json
  reduced_operator.npy
  basis.json
  couplings.json
  phase_sweeps/
```

The manifest records configuration, checkpoint, model identifier, selected parameter names, state dimension, projection residuals, moment evidence class, Python/PyTorch/device environment, and elapsed time.

## 10. Campaign analysis

Feature aggregation is deterministic and does not invent labels. Future-event labels and conventional diagnostics are joined in a separate governed step. Predictive evaluation uses grouped holdout by seed/run. A random row split is prohibited.

Variance analysis must distinguish at least step and seed effects, with batch, projection, and numerical repeats added in the full campaign.

## 11. Planner and continuation

The planner proposes rather than silently applies. Every proposal records:

- trigger and threshold;
- candidate set;
- selected candidate;
- predicted reserve;
- confidence and sensitivity;
- rollback condition.

Matched continuation begins from identical weights, optimizer state, data sequence, and RNG state. One control mechanism changes at a time. Current controls are learning-rate scale, beta1, denominator floor, and clipping. Exact historical replay requires native state plus original token stream and scheduler/overflow state.

## 12. Colab runner harness

Notebooks are canonical executable specifications. Shell wrappers use the official Colab CLI lifecycle:

```bash
colab new -s NAME --gpu T4
colab exec -s NAME -f notebook.ipynb
colab log -s NAME -o execution.ipynb
colab download -s NAME /content/cps-export artifacts/...
colab stop -s NAME
```

The stop operation is protected by a shell trap. Notebook outputs are not committed; exported evidence and logs are retrieved locally.

## 13. Test obligations

Unit and integration fixtures cover:

- magnitude preservation and perturbation norm;
- analytic 2x2 spectra;
- cycle-support invariance;
- singular-value preservation;
- branch tracking;
- finite-horizon gain;
- state flatten/unflatten;
- functional AdamW JVPs;
- native ZeRO state reconstruction;
- projected-operator recovery and closure;
- planner selection;
- feature and variance extraction;
- notebook syntax and cleared outputs.

CI runs Python 3.10 and 3.12 tests, notebook validation, syntax compilation, and package build.

## 14. Performance and escalation

Prototype target: analyze 24 couplings of a rank-64 reduced operator with 33 phases in under one minute, excluding JVP basis construction. Measurement cadence must be sparse enough to remain below two percent amortized training overhead before deployment as an online monitor.

Escalation from 70M to 160M/410M is permitted only after replay, derivative, projection, and predictive gates pass.
