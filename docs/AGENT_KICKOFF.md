# Codex Kickoff Prompt

You are implementing the first production-grade version of **Coupling-Phase Spectroscopy (CPS)**.

CPS is a structured pseudospectral probe of a projected optimizer-state Jacobian. It rotates selected couplings without changing their magnitude, then measures spectral migration, transient amplification, eigenvalue conditioning, mode collisions, and trace-moment harmonics. Its purpose is to predict fragile training dynamics and to plan damping, momentum, preconditioning, and optimizer block structure.

## Read first

1. `docs/SPEC.md`
2. `docs/METRICS.md`
3. `docs/RUN_MATRIX.md`
4. `docs/IMPLEMENTATION_BRIEF.md`
5. `manuscript/cps_preprint.tex`

## Immediate objective

Implement a PyTorch-functional optimizer-state analysis path for SGD, heavy-ball momentum, and AdamW, then reproduce Stages A and B of the run matrix.

## Non-negotiable contracts

- The optimizer step must be a deterministic pure function when batch and RNG state are frozen.
- The Jacobian is never materialized for neural systems; expose JVPs.
- Stateful optimizers include all state variables in the Jacobian.
- Reduced operators use float64 or complex128.
- Eigenvalues are tracked by continuation; independent sorting is prohibited.
- Every numerical claim records residuals, tolerances, phase grid, projection basis, and seed.
- Every experiment is reproducible from one manifest.
- Negative results are retained.

## Work packages

### WP1 — Functional optimizer maps

Implement flatten/unflatten utilities and functional steps for SGD, momentum, and AdamW. Write analytic quadratic fixtures.

### WP2 — JVP and projection

Implement `torch.func.jvp`, finite-difference verification, Arnoldi, and two-sided projection. Record Ritz residuals.

### WP3 — CPS families and observables

Complete block phase families, adaptive phase refinement, Schur fallback, trace-moment harmonics, and structured risk tables.

### WP4 — Exact benchmark suite

Run A1–A6 and B1–B4. Generate immutable artifacts and manuscript-ready figures.

### WP5 — Planning controllers

Implement auditable grid-search controllers for damping and momentum. No controller may apply an intervention without emitting its decision record.

## First acceptance test

For an isospectral normal/non-normal pair, demonstrate:

- equal nominal eigenvalues;
- unequal finite-horizon gain;
- CPS detection of phase-dependent path interference;
- exact agreement between dense and matrix-free reduced operators;
- deterministic reproduction from the saved manifest.

Do not expand to transformer training until all exact benchmarks pass.
