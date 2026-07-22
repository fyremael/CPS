from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable, Mapping


_LAYER_RE = re.compile(r"(?:gpt_neox\.)?layers\.(\d+)\.")


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    parameter_names: tuple[str, ...]
    numel: int
    layer: int | None
    subsystem: str


def subsystem_for_parameter(name: str) -> str:
    lower = name.lower()
    if "embed" in lower:
        return "embedding"
    if "query_key_value" in lower or ".attention." in lower:
        return "attention"
    if ".mlp." in lower or "dense_h_to_4h" in lower or "dense_4h_to_h" in lower:
        return "mlp"
    if "layernorm" in lower or "layer_norm" in lower or ".norm" in lower:
        return "normalization"
    if "embed_out" in lower or "lm_head" in lower:
        return "readout"
    return "other"


def layer_for_parameter(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def select_parameter_names(
    named_parameters: Mapping[str, object] | Iterable[tuple[str, object]],
    patterns: Iterable[str],
) -> tuple[str, ...]:
    items = named_parameters.items() if isinstance(named_parameters, Mapping) else named_parameters
    names = [name for name, _ in items]
    selected: list[str] = []
    for pattern in patterns:
        if pattern in names:
            selected.append(pattern)
            continue
        matches = [name for name in names if fnmatch(name, pattern) or pattern in name]
        selected.extend(matches)
    return tuple(dict.fromkeys(selected))


def build_parameter_blocks(named_parameters: Iterable[tuple[str, object]]) -> tuple[ParameterBlock, ...]:
    blocks: list[ParameterBlock] = []
    for name, parameter in named_parameters:
        numel = int(parameter.numel())  # type: ignore[attr-defined]
        blocks.append(
            ParameterBlock(
                name=name,
                parameter_names=(name,),
                numel=numel,
                layer=layer_for_parameter(name),
                subsystem=subsystem_for_parameter(name),
            )
        )
    return tuple(blocks)
