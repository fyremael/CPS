from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

from . import native_state as legacy
from . import native_state_grouped as grouped

_ORIGINAL_INFER = grouped._infer_optimizer_shape_groups


def _tensor_leaves(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Collect tensor leaves below a named optimizer-state field.

    Historical DeepSpeed packets are not uniform: an ``exp_avg`` field may be a
    tensor, a list of tensors, or a mapping of flattened groups. The original
    reader handled only the first representation.
    """

    import torch

    output: list[tuple[tuple[str, ...], Any]] = []
    if torch.is_tensor(value):
        output.append((path, value.detach().cpu().reshape(-1)))
    elif isinstance(value, Mapping):
        for child_key, child in value.items():
            output.extend(_tensor_leaves(child, (*path, str(child_key))))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            output.extend(_tensor_leaves(child, (*path, str(index))))
    return output


def _moment_tensors(payload: Mapping[str, Any], key: str) -> list[Any]:
    """Read all flattened moment groups, including list/mapping-valued fields."""

    output: list[tuple[tuple[str, ...], Any]] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                child_path = (*path, str(child_key))
                if str(child_key) == key:
                    leaves = _tensor_leaves(child)
                    output.extend(((*child_path, *leaf_path), tensor) for leaf_path, tensor in leaves)
                else:
                    visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(payload)
    output.sort(key=lambda item: item[0])
    return [tensor for _, tensor in output]


def _numel(group: OrderedDict[str, tuple[int, ...]]) -> int:
    return sum(legacy._numel(shape) for shape in group.values())


def _infer_optimizer_shape_groups(
    caller_groups: list[OrderedDict[str, tuple[int, ...]]],
    capacities: tuple[int, ...],
    *,
    partition_count: int,
) -> tuple[list[OrderedDict[str, tuple[int, ...]]], dict[str, Any] | None]:
    """Recover a released packet that exposes only one Adam moment group.

    The public Pythia-70M packet observed in Colab exposes one flattened moment
    group of 70,385,664 elements, while the matching Transformers model has
    70,426,624 parameters. The exact 40,960-element complement is the set of
    one-dimensional bias and LayerNorm tensors. CPS therefore permits a
    *partial native-state contract* only when one standard GPT-NeoX parameter
    class matches the observed capacity uniquely. Requests for uncovered
    parameters still fail closed.
    """

    if len(caller_groups) != 1 or len(capacities) != 1:
        return _ORIGINAL_INFER(
            caller_groups,
            capacities,
            partition_count=partition_count,
        )

    complete = caller_groups[0]
    decay = OrderedDict(
        (name, shape)
        for name, shape in complete.items()
        if grouped._uses_weight_decay(name, shape)
    )
    no_decay = OrderedDict(
        (name, shape)
        for name, shape in complete.items()
        if not grouped._uses_weight_decay(name, shape)
    )

    observed = capacities[0]
    maximum_padding = 2 * partition_count

    def fits(group: OrderedDict[str, tuple[int, ...]]) -> bool:
        padding = observed - _numel(group)
        return 0 <= padding <= maximum_padding

    if fits(complete):
        return caller_groups, None

    candidates: list[
        tuple[str, OrderedDict[str, tuple[int, ...]], OrderedDict[str, tuple[int, ...]]]
    ] = []
    if decay and fits(decay):
        candidates.append(("weight_decay_only", decay, no_decay))
    if no_decay and fits(no_decay):
        candidates.append(("no_decay_only", no_decay, decay))

    if len(candidates) != 1:
        sizes = {
            "complete": _numel(complete),
            "weight_decay_only": _numel(decay),
            "no_decay_only": _numel(no_decay),
        }
        if not candidates:
            raise ValueError(
                "single native Adam moment group does not match the complete, "
                "weight-decay, or no-decay GPT-NeoX parameter contract: "
                f"observed={observed}, candidate sizes={sizes}"
            )
        raise ValueError(
            "single native Adam moment group is ambiguous under the GPT-NeoX "
            f"parameter contract: observed={observed}, candidate sizes={sizes}"
        )

    coverage, available, unavailable = candidates[0]
    expected = _numel(available)
    report = {
        "method": "single_native_moment_group_capacity",
        "rule": "rank_gt_1_weight_decay_else_no_decay",
        "coverage": coverage,
        "group_count": 1,
        "expected_group_numel": [expected],
        "observed_group_numel": [observed],
        "padding_numel": [observed - expected],
        "parameter_counts": [len(available)],
        "maximum_allowed_padding_per_group": maximum_padding,
        "unavailable_parameter_class": (
            "one_dimensional_no_decay"
            if coverage == "weight_decay_only"
            else "matrix_and_embedding_weight_decay"
        ),
        "unavailable_parameter_count": len(unavailable),
        "unavailable_parameter_numel": _numel(unavailable),
        "partial_native_state": True,
    }
    return [available], report


# Install compatibility readers before exposing the reconstruction entry point.
legacy._moment_tensors = _moment_tensors
grouped._infer_optimizer_shape_groups = _infer_optimizer_shape_groups
reconstruct_zero_adam_state = grouped.reconstruct_zero_adam_state

__all__ = ["reconstruct_zero_adam_state"]
