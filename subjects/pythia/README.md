# Pythia subject adapter

This directory is the empirical subject layer for Coupling-Phase Spectroscopy (CPS). The executable package lives under `src/cps/pythia`; this directory holds governed configurations, checkpoint registries, and Colab runner contracts.

## Subject hierarchy

1. **Pythia-70M**: implementation, dense longitudinal study, and native optimizer-state reconstruction.
2. **PolyPythia-70M**: seed variance and prospective outlier prediction.
3. **PolyPythia-160M causal variants**: separation of initialization and data-order effects.
4. **Pythia-160M/410M**: first scale-transfer gate.
5. **Pythia-1B/2.8B**: sparse external validation after the 70M/160M/410M gates pass.

## Exact versus reconstructed optimizer state

`moment_source: native` invokes the offline ZeRO state reconstructor against a downloaded GPT-NeoX checkpoint. `moment_source: reconstructed` initializes the first moment to zero and the second moment to the configured floor. Reconstructed-state runs are instrumentation and architecture studies; they must not be presented as exact historical training-state measurements.

## Run contracts

Every run writes:

- `manifest.json`: complete configuration, environment, model revision, selected coordinates, and projection diagnostics;
- `reduced_operator.npy`: projected optimizer-state Jacobian;
- `basis.json`: semantic basis registry;
- `couplings.json`: ranked coupling-phase measurements.

No intervention claim is accepted without a matched continuation run on the same subsequent token sequence.
