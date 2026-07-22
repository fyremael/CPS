from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FiniteDifferenceConfig:
    relative_step: float = 1e-5
    absolute_floor: float = 1e-7


def finite_difference_jvp(
    update_map: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    state: NDArray[np.float64],
    direction: NDArray[np.float64],
    config: FiniteDifferenceConfig = FiniteDifferenceConfig(),
) -> NDArray[np.float64]:
    """Matrix-free JVP of a deterministic, frozen-randomness optimizer map."""

    state = np.asarray(state, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if state.shape != direction.shape:
        raise ValueError("state and direction must have the same shape")
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros_like(state)
    scale = max(np.linalg.norm(state), 1.0)
    step = max(config.absolute_floor, config.relative_step * scale / norm)
    plus = np.asarray(update_map(state + step * direction), dtype=float)
    minus = np.asarray(update_map(state - step * direction), dtype=float)
    if plus.shape != state.shape or minus.shape != state.shape:
        raise ValueError("update_map must preserve state shape")
    return (plus - minus) / (2.0 * step)


def torch_functional_jvp(function: Callable[..., Any], primals: tuple[Any, ...], tangents: tuple[Any, ...]):
    """Optional PyTorch JVP wrapper, imported lazily."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for torch_functional_jvp") from exc
    return torch.func.jvp(function, primals, tangents)
