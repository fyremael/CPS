"""Pythia subject adapter for Coupling-Phase Spectroscopy."""

from .config import PythiaProbeConfig, load_probe_config
from .registry import PythiaRunSpec, get_run_spec, list_run_specs

# Historical Pythia GPT-NeoX packets can omit ``param_shapes`` while retaining
# separate flattened optimizer groups. Patch the native-state module before any
# runner imports it so direct and indirect callers share the governed grouped
# reconstruction path.
from . import native_state as _native_state
from .native_state_grouped import reconstruct_zero_adam_state as _grouped_reconstruct

_native_state.reconstruct_zero_adam_state = _grouped_reconstruct

__all__ = [
    "PythiaProbeConfig",
    "PythiaRunSpec",
    "get_run_spec",
    "list_run_specs",
    "load_probe_config",
]
