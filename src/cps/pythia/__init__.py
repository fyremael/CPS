"""Pythia subject adapter for Coupling-Phase Spectroscopy."""

from .config import PythiaProbeConfig, load_probe_config
from .registry import PythiaRunSpec, get_run_spec, list_run_specs

# Historical Pythia GPT-NeoX packets vary in both metadata and optimizer-state
# layout. Install the compatibility reader before runner.py binds the native
# reconstruction function so direct and indirect callers share one governed
# path.
from . import native_state as _native_state
from .native_state_packet import reconstruct_zero_adam_state as _packet_reconstruct

_native_state.reconstruct_zero_adam_state = _packet_reconstruct

__all__ = [
    "PythiaProbeConfig",
    "PythiaRunSpec",
    "get_run_spec",
    "list_run_specs",
    "load_probe_config",
]
