# Pythia empirical protocol

## 1. Purpose

Pythia is the primary real-world subject for CPS because it supplies longitudinal model checkpoints, controlled data order, and—in selected native repositories—optimizer-state shards. PolyPythia supplies replicated seeds and outlier cases.

The programme tests whether structured coupling sensitivity predicts and helps prevent training disturbances beyond cheaper diagnostics.

## 2. Evidence classes

### E0 — synthetic

Analytic or generated matrices. Validates mathematics and software only.

### E1 — real weights, reconstructed moments

A real Pythia checkpoint with Adam moments initialized by a declared reconstruction rule. Validates the architecture-dependent loss and gradient map. It does not reproduce the historical optimizer state.

### E2 — real weights, native moments, surrogate batch

Native optimizer moments are reconstructed from GPT-NeoX ZeRO shards, but the probe batch is not the exact historical next batch. Validates optimizer memory while leaving a data-state mismatch.

### E3 — native state and reconstructed historical batch

Weights, optimizer state, schedule, and exact next token batch match the original run. This is the minimum evidence class for claims about the historical local training dynamics.

### E4 — matched continuation intervention

A preregistered CPS recommendation is tested against a matched baseline fork using the same initial state and subsequent token sequence.

## 3. Primary checkpoint grid

Pilot:

```text
step0, step1, step2, step4, step16, step128, step512,
step1000, step2000, step4000, step8000, step16000,
step32000, step64000, step128000, step143000
```

The dense 154-checkpoint run begins only after projection-rank, phase-grid, and batch-repeatability gates pass.

## 4. Coordinate contract

The implemented state is

\[
(\theta,m,u),\qquad u=\log(v+\tau).
\]

A diagonal similarity scaling normalizes parameter and first-moment coordinates by block RMS. The log-second-moment coordinate is dimensionless. Every reported operator is therefore tied to a declared state chart and semantic block basis.

## 5. Projection contract

The basis is constructed from named parameter blocks and declared components: parameter direction, optimizer update, first-moment direction, second-moment variation, and seeded random probes. Modified Gram–Schmidt produces an orthonormal basis in scaled state coordinates.

For every source vector, record

\[
r_j=\frac{\|Jq_j-QQ^*Jq_j\|}{\|Jq_j\|}.
\]

High closure residuals invalidate strong interpretations of the reduced operator and trigger rank or basis escalation.

## 6. Prospective prediction

At checkpoint \(t\), CPS features are computed without access to outcomes after \(t\). Future-event labels are defined on a preregistered horizon. Splits are by seed or model scale, not by randomly shuffling adjacent checkpoints.

The decisive comparison is:

```text
conventional diagnostics
versus
conventional diagnostics + CPS
```

The principal gate is a held-out AUC improvement of at least 0.05, with positive warning lead time and stability across projection seeds.

## 7. Intervention

Recommendations are hypotheses. Each is tested in a matched continuation with identical initial state, token sequence, precision, and compute allowance. One control family is changed at a time.

Initial controls:

- learning-rate scale;
- first-moment coefficient;
- second-moment coefficient;
- Adam denominator floor;
- gradient clipping;
- block-local variants after the scalar controls are understood.

## 8. Failure conditions

CPS is rejected as an independent planning signal when:

- nominal finite-horizon gain explains the same events;
- results change materially under harmless basis perturbations;
- projection closure remains poor at feasible rank;
- native-state and reconstructed-state conclusions disagree without a stable correction;
- recommendations fail matched continuations;
- measurement cost exceeds its warning or intervention value.
