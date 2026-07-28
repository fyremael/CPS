# Tran–Vu moderate-gap characterization

This package characterizes the new CPS subspace-stability certificate as a research instrument rather than merely checking that its functions return values.

It answers four questions:

1. When are the Tran–Vu hypotheses satisfied?
2. When is the refined bound informative and sharper than classical Davis–Kahan?
3. How do directional coupling, eigengap, perturbation size, and cluster breadth control the result?
4. Which conclusions remain outside the certificate, especially for non-normal operators?

The study is deterministic, CPU-only, and self-contained once CPS is installed. It does not download a model or require a previous notebook.

## Run

From the repository root:

```bash
python -m pip install -e '.[dev,notebooks]'
python -m experiments.tran_vu.run
```

A smaller execution path, useful for development and CI, preserves every dataset and figure class:

```bash
python -m experiments.tran_vu.run --quick
```

Choose a different output directory with:

```bash
python -m experiments.tran_vu.run --output /path/to/results
```

The runner exits nonzero when an acceptance gate fails. `--allow-failures` is available only for exploratory inspection of a failed packet.

## Governed regime matrix

The package includes ten named fixtures:

| Regime | Intended lesson |
|---|---|
| `weak_directional_coupling` | The perturbation is large globally but misses the nearby signal block; Tran–Vu should improve on Davis–Kahan. |
| `classical_sharper_large_gap` | The ordinary large-gap bound is already tighter. |
| `valid_but_vacuous` | The theorem applies but its bound exceeds one. |
| `gap_failure` | The perturbation is too large relative to the local gap. |
| `signal_scale_failure` | The gap is not moderate relative to the signal singular value. |
| `strong_local_coupling` | Direct coupling inside the sensitive block destroys the directional advantage. |
| `high_halving_rank_penalty` | A broad near-signal cluster exposes the quadratic penalty in the halving rank. |
| `complex_realification` | Complex phase perturbations are represented by an equivalent real block matrix. |
| `nonnormal_singular_stability` | Singular-space stability remains meaningful while eigenvector sensitivity is kept separate. |
| `rank_two_cluster` | The target may be a multi-dimensional leading subspace rather than one vector. |

Each fixture declares its expected theorem applicability, informativeness, comparison with Davis–Kahan, admission state, required failure reason, representation, and minimum halving rank where relevant. A mismatch fails the package.

## Parameter sweeps

The package then varies one mechanism at a time:

- directional coupling ratio `x / ||E||` at fixed perturbation norm;
- gap-to-noise ratio `δₚ / ||E||`;
- perturbation-to-gap ratio `||E|| / δₚ`;
- halving rank `r`;
- a two-dimensional admission map over gap and directional coupling;
- non-normality while preserving the singular-spectrum construction;
- random complex matrices for realification validation.

Every sweep is written in both CSV and JSON form.

## Obligatory visual outputs

A successful run produces PNG and SVG versions of all nine figures:

```text
figures/
├── 01_regime_bound_comparison.{png,svg}
├── 02_directional_coupling_sweep.{png,svg}
├── 03_gap_sweep.{png,svg}
├── 04_perturbation_sweep.{png,svg}
├── 05_halving_rank_penalty.{png,svg}
├── 06_admission_map.{png,svg}
├── 07_observed_vs_bounds.{png,svg}
├── 08_nonnormality_separation.{png,svg}
└── 09_realification_validation.{png,svg}
```

The package also writes `index.html`, a standalone visual report containing the regime table, acceptance gates, and all plots. Open that file directly in a browser; it has no external web dependencies.

## Complete output packet

The default packet is written to `artifacts/tran_vu_characterization`:

```text
acceptance_report.json
admission_map.{csv,json}
directional_coupling_sweep.{csv,json}
gap_sweep.{csv,json}
halving_rank_sweep.{csv,json}
index.html
manifest.json
nonnormality_sweep.{csv,json}
perturbation_sweep.{csv,json}
realification_validation.{csv,json}
regime_matrix.{csv,json}
report.json
figures/
```

`manifest.json` records SHA-256 hashes and byte sizes for every generated artifact.

## Acceptance gates

The governed contract is in `acceptance.json`. A run passes only when:

- every fixture matches its declared outcome;
- every applicable Tran–Vu bound contains the measured projector displacement;
- every admitted result is informative and sharper than Davis–Kahan;
- all ten regimes are present;
- all nine PNG and nine SVG figures exist;
- realification preserves the operator norm and duplicated singular spectrum to the declared tolerance;
- the standalone HTML report exists.

## Reading the plots

The central visual is the admission map. The horizontal coordinate measures how separated the target singular space is relative to perturbation size. The vertical coordinate measures how directly the perturbation couples into the nearby signal block. The admitted region should be a narrow band: the theorem is useful when the gap is moderate and the local coupling is unusually weak.

The non-normality figure is intentionally separate. It demonstrates that a singular-space certificate does not become an eigenvector or pseudospectral certificate merely because the bound is admitted.

## Notebook

`notebooks/07_tran_vu_characterization.ipynb` provides the same experiment as a release-grade Colab lesson. It executes the package from a fresh runtime, displays the generated report and all figures, and exports the evidence through the common CPS notebook workflow.

## Evidence boundary

This package establishes implementation correctness and synthetic characterization of the certificate. It does not establish that CPS predicts training instability, that a projected operator faithfully captures the full optimizer state, or that a certified singular subspace prevents transient amplification. Those require separate empirical and pseudospectral evidence.
