from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TensorSlice:
    name: str
    shape: tuple[int, ...]
    theta: slice
    momentum: slice
    log_second_moment: slice
    numel: int


@dataclass(frozen=True)
class StateLayout:
    tensors: tuple[TensorSlice, ...]
    parameter_numel: int
    state_numel: int

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "StateLayout":
        offset = 0
        raw: list[tuple[str, tuple[int, ...], int, slice]] = []
        for name, tensor in parameters.items():
            shape = tuple(int(x) for x in tensor.shape)  # type: ignore[attr-defined]
            numel = int(tensor.numel())  # type: ignore[attr-defined]
            raw.append((name, shape, numel, slice(offset, offset + numel)))
            offset += numel
        parameter_numel = offset
        tensors = tuple(
            TensorSlice(
                name=name,
                shape=shape,
                theta=theta,
                momentum=slice(theta.start + parameter_numel, theta.stop + parameter_numel),
                log_second_moment=slice(
                    theta.start + 2 * parameter_numel,
                    theta.stop + 2 * parameter_numel,
                ),
                numel=numel,
            )
            for name, shape, numel, theta in raw
        )
        return cls(tensors=tensors, parameter_numel=parameter_numel, state_numel=3 * parameter_numel)

    def tensor(self, name: str) -> TensorSlice:
        for entry in self.tensors:
            if entry.name == name:
                return entry
        raise KeyError(name)
