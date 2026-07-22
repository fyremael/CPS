# Result schema

## `manifest.json`

Required fields:

- `schema_version`;
- `subject`, `run_spec`, `model_id`, `revision`;
- selected parameter names and total element count;
- state dimension and basis rank;
- projection closure diagnostics and JVP norms;
- native-state evidence class and source files;
- full run configuration;
- Python, PyTorch, device and dtype metadata;
- elapsed time.

## `basis.json`

Each entry contains a stable index, human-readable name, original parameter name, and state component.

## `couplings.json`

Each record declares source and target basis coordinates, nominal coupling magnitude, perturbation-family metadata, and the complete CPS metric vector.

## Versioning

Fields may be added within a schema version. Removing or changing the meaning of a field requires a new `schema_version` and a migration note.
