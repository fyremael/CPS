# Work Packages

Status values: **implemented**, **implemented with empirical execution pending**, or **research extension**.

## WP0 — Mathematical closure

**Owner:** Axiomatist + Formalist  
**Status:** implemented

Delivered:

- scalar and singular-channel magnitude-preserving families;
- perturbation-norm and ordinary-pseudospectrum relation;
- cycle-support and trace-harmonic framing;
- basis-dependence taxonomy;
- frozen-step scope statement;
- analytic and numerical fixtures.

Remaining research extension: a complete block-covariance theorem and sharper projection-error bounds for nonnormal reduced operators.

## WP1 — Functional optimizer-state maps

**Owner:** Steward + Verifier  
**Status:** implemented for selected-coordinate AdamW; extensions remain

Delivered:

- flat theta/m/log-v state layout;
- differentiable selected-coordinate AdamW map;
- explicit frozen batch;
- autodiff and centered finite-difference JVP paths;
- dimensionless diagonal coordinate scaling;
- native/reconstructed moment provenance.

Research extensions: full-model functionalization, SGD/heavy-ball subject adapters, exact GPT-NeoX scheduler/overflow state, distributed optimizer-state replay.

## WP2 — Projection and numerical spectral engine

**Owner:** Cartographer + Verifier  
**Status:** implemented

Delivered:

- semantic basis construction;
- matrix-free projected Jacobian;
- projection closure residuals;
- scalar and singular-channel phase sweeps;
- eigenpair continuation;
- spectral radius, finite-horizon gain, gaps, conditioning, loop geometry, Kreiss surrogate;
- campaign artifact schema and tests.

Research extensions: two-sided neural projection, adaptive phase refinement, Schur continuation near defective clusters.

## WP3 — Pythia subject adapter

**Owner:** Measurement Minder + Steward  
**Status:** implemented with empirical execution pending

Delivered:

- governed Pythia/PolyPythia registry;
- Hugging Face checkpoint loading;
- native GPT-NeoX checkpoint download and inspection;
- ZeRO-partitioned Adam moment reconstruction;
- deterministic probe batches;
- checkpoint probe and longitudinal runners;
- feature aggregation, grouped prediction, and variance decomposition;
- run configurations and evidence manifests.

Exit evidence requires completed P0/P1 Colab packets. Code availability is not empirical validation.

## WP4 — Planning controllers

**Owner:** Judgment Minder + Adversary  
**Status:** implemented as proposal and matched-test harness

Delivered:

- scalar candidate planner;
- damping-family evaluation;
- learning-rate, momentum, denominator-floor, and clipping continuation controls;
- identical-start matched continuation recorder;
- planner proposal schema.

Research extensions: block merge/split optimizer realization, Shampoo/Muon adapters, policy confidence calibration.

## WP5 — Colab runner fabric

**Owner:** Composer + Steward  
**Status:** implemented

Delivered:

- seven executable notebooks;
- official Colab CLI lifecycle wrapper;
- T4/A100 runner scripts;
- runtime cleanup trap;
- execution-log and artifact retrieval;
- notebook syntax/output validation.

Operational prerequisite: local Colab CLI authentication and available compute units. No accelerator execution is claimed by repository construction alone.

## WP6 — Empirical Pythia campaign

**Owner:** Measurement Minder + Referee  
**Status:** execution pending

Required sequence:

1. P0 smoke, replay, and JVP validation;
2. P1 Pythia-70M longitudinal campaign;
3. P2 PolyPythia seed study and prospective outlier test;
4. P3 initialization/data-order attribution;
5. P4 scale transfer;
6. matched continuation interventions.

Each gate produces immutable packets, checksums, logs, decision records, and a referee acceptance report.

## WP7 — Preprint evidence revision

**Owner:** Amanuensis + Referee  
**Status:** programme manuscript updated; evidence revision pending

Delivered:

- formal instrument and propositions;
- Pythia longitudinal laboratory protocol;
- implementation architecture and limitations;
- executable repository references.

Remaining:

- insert completed empirical tables and figures;
- report null results and failed interventions;
- add immutable artifact identifiers;
- conduct council and external referee review;
- prepare submission version only after evidence gates pass.
