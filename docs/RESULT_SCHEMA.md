# Result schema

## `manifest.json`

Required fields:

- `schema_version`;
- `subject`, `run_spec`, `model_id`, `revision`;
- `run_spec.key` as the stable registry identifier;
- selected parameter names and total element count;
- state dimension and basis rank;
- projection closure diagnostics and JVP norms;
- requested and active attention implementations;
- requested and effective JVP modes/backends, fallback flag, fallback reason, and preflight norm;
- native-state evidence class and source files;
- full run configuration;
- Python, PyTorch, device and dtype metadata;
- elapsed time.

`run_spec` may also include `family`, `seed`, `weight_seed`, `data_seed`, and notes. Notebook readers may accept a legacy `run_spec.name` field, but newly written schema-v2 evidence uses `run_spec.key`.

## `basis.json`

Each entry contains a stable index, human-readable name, original parameter name, and state component.

## `couplings.json`

Each record declares source and target basis coordinates, nominal coupling magnitude, perturbation-family metadata, and the complete CPS metric vector.

## Versioning

Fields may be added within a schema version. Removing or changing the meaning of a field requires a new `schema_version` and a migration note.

### Schema version 2 additions

`manifest.json` now contains:

```text
attention.requested_implementation
attention.active_implementation
jacobian.requested_mode
jacobian.requested_backend
jacobian.effective_mode
jacobian.effective_backend
jacobian.fallback_used
jacobian.fallback_reason
jacobian.preflight_norm
```

These fields prevent an eager-forward-AD run and a centered finite-difference fallback from being pooled as though they measured the reduced operator by the same numerical method.
