# Metric Contracts

Each metric must define its operator, perturbation family, phase grid, projection basis, horizon, and numerical tolerance. A scalar without these fields is not a CPS measurement.

## 1. Nominal operator fields

- `checkpoint_step`
- `optimizer_name`
- `optimizer_hyperparameters`
- `projection_method`
- `projection_rank`
- `basis_seed`
- `state_partition`
- `batch_fingerprint`
- `randomness_fingerprint`

## 2. Coupling fields

- `target_block`
- `source_block`
- `family_type`: `entry_phase`, `real_pair`, `block_svd_phase`, or `basis_rotation`
- `baseline_magnitude`
- `phase_budget`
- `phase_count`
- `perturbation_norm_max`
- `cycle_supported`: whether the coupling belongs to a directed cycle of the reduced support graph

## 3. Primary metrics

### `spectral_radius_max`

Worst modulus of any eigenvalue over the sweep. For discrete-time local maps, values above one indicate a possible expanding mode.

### `spectral_abscissa_max`

Worst real part over the sweep. Use for continuous-time generators or logarithms of update maps.

### `finite_horizon_gain`

\[
G(K)=\max_{\phi}\max_{1\leq k\leq K}\|A(\phi)^k\|_2.
\]

Report the maximizing phase and horizon.

### `kreiss_surrogate`

Grid approximation to the discrete Kreiss constant. The grid must be included in run metadata. It is a comparative indicator, not a certified value.

### `minimum_gap`

Minimum pairwise eigenvalue separation. Report branch-tracking confidence near the minimizer.

### `maximum_eigenvalue_condition`

Maximum left-right eigenvector condition number. Infinite or saturated values must not be silently clipped.

### `loop_displacement`

Maximum matched-branch displacement from baseline. This depends on eigenpair tracking and must be accompanied by a branch-invariant metric.

### `trace_moment_harmonics`

Fourier coefficients of \(\operatorname{tr}(A(\phi)^q)\) for orders \(q=1,\ldots,q_{\max}\). These are the preferred branch-invariant spectral response features.

## 4. Composite risk

A default exploratory score is

\[
\mathcal R_{ij}
=R_{ij}
+w_G\log(1+G_{ij})
+w_\kappa\log(1+\kappa_{ij}^{\max})
+\frac{w_\Delta}{\Delta_{ij}+\epsilon}.
\]

The weights are not universal constants. They must be calibrated on a development split and frozen before final evaluation.

## 5. Prediction targets

- loss spike within \(H\) steps;
- gradient norm exceeding a predeclared threshold;
- optimizer-state norm spike;
- skipped or overflowed mixed-precision step;
- divergence or NaN;
- abrupt degradation in validation loss;
- recovery time after a bounded stress injection.

## 6. Reporting

Every result table must include:

- mean and confidence interval over seeds;
- probe-rank sensitivity;
- phase-grid sensitivity;
- checkpoint cadence;
- measurement overhead;
- strongest baseline;
- intervention cost;
- negative or null results.
