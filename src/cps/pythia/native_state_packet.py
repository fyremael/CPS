"""Compatibility export for the governed native optimizer-state reader.

This module is intentionally side-effect free. Earlier versions monkey-patched
``native_state_grouped`` and captured the previous implementation in a module
global. Reloading the module in Colab could therefore make the fallback call
itself recursively. The stable reader now owns all packet-layout logic directly.
"""

from .native_state_stable import reconstruct_zero_adam_state

__all__ = ["reconstruct_zero_adam_state"]
