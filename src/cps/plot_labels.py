from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_PARAMETER_REPLACEMENTS = (
    (".attention.query_key_value.weight", ".attn.qkv"),
    (".attention.dense.weight", ".attn.out"),
    (".mlp.dense_h_to_4h.weight", ".mlp.up"),
    (".mlp.dense_4h_to_h.weight", ".mlp.down"),
    (".input_layernorm.weight", ".ln.in"),
    (".post_attention_layernorm.weight", ".ln.post"),
    (".final_layer_norm.weight", ".ln.final"),
    (".embed_in.weight", ".embed.in"),
    (".embed_out.weight", ".embed.out"),
)

_COMPONENT_LABELS = {
    "theta": "θ",
    "update": "Δθ",
    "gradient": "g",
    "moment": "m",
    "exp_avg": "m",
    "variance": "v",
    "exp_avg_sq": "v",
}


def compact_parameter_name(name: str) -> str:
    """Produce a short, architecture-aware block label without losing layer identity."""

    compact = name.removeprefix("module.").removeprefix("gpt_neox.")
    parts = compact.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        compact = f"L{parts[1]}." + ".".join(parts[2:])
    for source, target in _PARAMETER_REPLACEMENTS:
        compact = compact.replace(source, target)
    return compact


def compact_component_name(name: str) -> str:
    if name in _COMPONENT_LABELS:
        return _COMPONENT_LABELS[name]
    if name.startswith("random"):
        suffix = name.removeprefix("random")
        if suffix.isdigit():
            return "r" + suffix.translate(_SUBSCRIPT)
    return name


def _deduplicate(labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    counts = Counter(labels)
    seen: Counter[str] = Counter()
    output: list[str] = []
    for label in labels:
        seen[label] += 1
        if counts[label] == 1:
            output.append(label)
        else:
            output.append(f"{label}·{seen[label]}")
    return output


def compact_basis_labels(
    basis: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], str | None, list[tuple[str, str]]]:
    """Return plot labels, full-to-short mapping, shared block, and an exact label key."""

    full_names = [str(item["name"]) for item in basis]
    parsed: list[tuple[str, str]] = []
    for full_name in full_names:
        parameter, separator, component = full_name.rpartition(":")
        if not separator:
            parameter, component = full_name, ""
        parsed.append((compact_parameter_name(parameter), compact_component_name(component)))

    blocks = [block for block, _ in parsed]
    components = [component for _, component in parsed]
    shared_block = blocks[0] if blocks and len(set(blocks)) == 1 and all(components) else None

    if shared_block is not None:
        labels = components
    else:
        labels = [f"{block} · {component}" if component else block for block, component in parsed]
    labels = _deduplicate(labels)
    mapping = dict(zip(full_names, labels, strict=True))
    key = list(zip(labels, full_names, strict=True))
    return labels, mapping, shared_block, key
