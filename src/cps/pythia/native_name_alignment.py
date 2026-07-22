from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_TRANSFORMER_LAYER = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class NativeNameAlignment:
    requested_to_native: dict[str, str]
    transforms: dict[str, str]
    report: dict[str, Any]


def _flatten_groups(
    groups: Sequence[Mapping[str, Sequence[int]]],
) -> OrderedDict[str, tuple[int, ...]]:
    output: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    for group in groups:
        for name, shape in group.items():
            if name in output:
                raise ValueError(f"duplicate parameter name in shape contract: {name}")
            output[str(name)] = tuple(int(value) for value in shape)
    return output


def _coerce_caller_shapes(
    value: Mapping[str, Sequence[int]]
    | Sequence[Mapping[str, Sequence[int]]]
    | None,
) -> OrderedDict[str, tuple[int, ...]]:
    if value is None:
        return OrderedDict()
    if isinstance(value, Mapping):
        value = [value]
    return _flatten_groups(value)


def _shape_transform(
    native_shape: tuple[int, ...],
    requested_shape: tuple[int, ...] | None,
) -> str | None:
    if requested_shape is None or native_shape == requested_shape:
        return "identity"
    if (
        len(native_shape) == 2
        and len(requested_shape) == 2
        and native_shape == tuple(reversed(requested_shape))
    ):
        return "transpose_2d"
    return None


def _direct_candidates(name: str) -> tuple[str, ...]:
    stripped = name.removeprefix("gpt_neox.")
    return (
        name,
        f"module.{name}",
        stripped,
        f"module.{stripped}",
        f"sequential.{stripped}",
        f"module.sequential.{stripped}",
    )


def _matching_tail(
    native_names: Sequence[str],
    native_shapes: Mapping[str, tuple[int, ...]],
    *,
    tail: str,
    requested_shape: tuple[int, ...] | None,
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for native_name in native_names:
        if native_name == tail or native_name.endswith(f".{tail}"):
            transform = _shape_transform(native_shapes[native_name], requested_shape)
            if transform is not None:
                matches.append((native_name, transform))
    return matches


def _layer_occurrence(
    requested_name: str,
    caller_shapes: Mapping[str, tuple[int, ...]],
    tail: str,
) -> int:
    parsed = _TRANSFORMER_LAYER.match(requested_name)
    if parsed is None:
        raise ValueError(f"not a GPT-NeoX transformer-layer parameter: {requested_name}")
    requested_layer = int(parsed.group(1))
    siblings: list[tuple[int, str]] = []
    for caller_name in caller_shapes:
        candidate = _TRANSFORMER_LAYER.match(caller_name)
        if candidate is not None and candidate.group(2) == tail:
            siblings.append((int(candidate.group(1)), caller_name))
    siblings.sort()
    for index, (layer, name) in enumerate(siblings):
        if layer == requested_layer and name == requested_name:
            return index
    # The selected name can be supplied without a complete caller contract.
    return requested_layer


def _special_candidates(
    requested_name: str,
    native_names: Sequence[str],
    native_shapes: Mapping[str, tuple[int, ...]],
    requested_shape: tuple[int, ...] | None,
) -> list[tuple[str, str, str]]:
    patterns: tuple[str, ...]
    selection = "unique"
    if requested_name == "gpt_neox.embed_in.weight":
        patterns = ("word_embeddings.weight", "embed_in.weight")
        selection = "first"
    elif requested_name == "embed_out.weight":
        patterns = ("final_linear.weight", "embed_out.weight", "word_embeddings.weight")
        selection = "last"
    elif requested_name.startswith("gpt_neox.final_layer_norm."):
        component = requested_name.rsplit(".", 1)[-1]
        patterns = (
            f"final_layer_norm.{component}",
            f"final_norm.{component}",
            f"norm.{component}",
        )
        selection = "last"
    else:
        return []

    matches: list[tuple[str, str, str]] = []
    for native_name in native_names:
        if any(native_name == pattern or native_name.endswith(f".{pattern}") for pattern in patterns):
            transform = _shape_transform(native_shapes[native_name], requested_shape)
            if transform is not None:
                matches.append((native_name, transform, selection))
    return matches


def resolve_requested_parameter_names(
    requested_names: Iterable[str] | None,
    native_groups: Sequence[Mapping[str, Sequence[int]]],
    parameter_shapes: Mapping[str, Sequence[int]]
    | Sequence[Mapping[str, Sequence[int]]]
    | None,
) -> NativeNameAlignment:
    """Align Hugging Face GPT-NeoX names with native pipeline-checkpoint names.

    Native GPT-NeoX/DeepSpeed metadata can use pipeline-stage prefixes while the
    loaded Transformers model uses names such as
    ``gpt_neox.layers.0.attention.dense.weight``. Alignment is accepted only
    when an exact candidate, a unique semantic suffix, or the same layer
    occurrence within a semantic suffix family also satisfies the shape
    contract. Ambiguous mappings fail closed.
    """

    if requested_names is None:
        return NativeNameAlignment({}, {}, {"method": "not_requested", "aliases": []})

    requested = tuple(dict.fromkeys(str(name) for name in requested_names))
    native_shapes = _flatten_groups(native_groups)
    native_names = list(native_shapes)
    caller_shapes = _coerce_caller_shapes(parameter_shapes)

    requested_to_native: dict[str, str] = {}
    transforms: dict[str, str] = {}
    aliases: list[dict[str, Any]] = []
    used_native: set[str] = set()

    for requested_name in requested:
        requested_shape = caller_shapes.get(requested_name)
        resolved: tuple[str, str, str] | None = None

        for candidate in _direct_candidates(requested_name):
            if candidate not in native_shapes:
                continue
            transform = _shape_transform(native_shapes[candidate], requested_shape)
            if transform is not None:
                resolved = (candidate, transform, "exact_or_prefix_variant")
                break

        parsed = _TRANSFORMER_LAYER.match(requested_name)
        if resolved is None and parsed is not None:
            tail = parsed.group(2)
            matches = _matching_tail(
                native_names,
                native_shapes,
                tail=tail,
                requested_shape=requested_shape,
            )
            if len(matches) == 1:
                native_name, transform = matches[0]
                resolved = (native_name, transform, "unique_semantic_suffix")
            elif matches:
                occurrence = _layer_occurrence(requested_name, caller_shapes, tail)
                if occurrence < len(matches):
                    native_name, transform = matches[occurrence]
                    resolved = (
                        native_name,
                        transform,
                        "semantic_suffix_layer_occurrence",
                    )

        if resolved is None:
            special = _special_candidates(
                requested_name,
                native_names,
                native_shapes,
                requested_shape,
            )
            if special:
                selection = special[0][2]
                if selection == "first":
                    native_name, transform, _ = special[0]
                elif selection == "last":
                    native_name, transform, _ = special[-1]
                elif len(special) == 1:
                    native_name, transform, _ = special[0]
                else:
                    native_name = ""
                    transform = "identity"
                if native_name:
                    resolved = (native_name, transform, f"special_{selection}")

        if resolved is None:
            tail = requested_name.rsplit(".", 2)[-2:]
            tail_text = ".".join(tail)
            nearby = [
                name
                for name in native_names
                if name.endswith(tail_text)
                and _shape_transform(native_shapes[name], requested_shape) is not None
            ][:12]
            raise KeyError(
                "could not align requested Transformers parameter with native "
                f"checkpoint metadata: requested={requested_name!r}, "
                f"requested_shape={requested_shape}, nearby_native_names={nearby}"
            )

        native_name, transform, method = resolved
        if native_name in used_native:
            other = next(
                key for key, value in requested_to_native.items() if value == native_name
            )
            raise ValueError(
                "native parameter name alignment is not one-to-one: "
                f"{other!r} and {requested_name!r} both map to {native_name!r}"
            )
        used_native.add(native_name)
        requested_to_native[requested_name] = native_name
        transforms[requested_name] = transform
        aliases.append(
            {
                "requested_name": requested_name,
                "native_name": native_name,
                "method": method,
                "requested_shape": (
                    None if requested_shape is None else list(requested_shape)
                ),
                "native_shape": list(native_shapes[native_name]),
                "transform": transform,
            }
        )

    return NativeNameAlignment(
        requested_to_native=requested_to_native,
        transforms=transforms,
        report={
            "method": "governed_gpt_neox_name_alignment",
            "requested_count": len(requested),
            "alias_count": sum(
                item["requested_name"] != item["native_name"] for item in aliases
            ),
            "aliases": aliases,
        },
    )


def apply_name_transform(value: Any, transform: str) -> Any:
    if transform == "identity":
        return value
    if transform == "transpose_2d":
        return value.transpose(0, 1).contiguous()
    raise ValueError(f"unsupported native parameter transform: {transform}")


__all__ = [
    "NativeNameAlignment",
    "apply_name_transform",
    "resolve_requested_parameter_names",
]
