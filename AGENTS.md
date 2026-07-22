# Agent operating contract

CPS is a measurement programme, not a plot-generating exercise.

Before modifying the repository:

1. Read `docs/SPEC.md`, `docs/METRICS.md`, and `docs/PYTHIA_PROTOCOL.md`.
2. Preserve the distinction between native and reconstructed optimizer state.
3. Keep minibatches, randomness, coordinate scaling, and projection bases explicit in manifests.
4. Add a unit test for every mathematical or checkpoint-format assumption.
5. Do not claim predictive or causal success from synthetic or smoke-test results.
6. Use a branch and pull request. Record unresolved obligations in `docs/COUNCIL_REVIEW.yaml`.

A contribution is complete only when tests pass, notebooks contain no committed outputs, evidence schemas remain backward compatible or are versioned, and the manuscript does not overstate implementation status.
