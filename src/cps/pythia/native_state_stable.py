from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import native_state as legacy


def _tensor_leaves(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], Any]]:
    """Collect flattened tensor leaves from a heterogeneous state field."""

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
    """Read tensor-, sequence-, or mapping-valued Adam moment groups."""

    output: list[tuple[tuple[str, ...], Any]] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                child_path = (*path, str(child_key))
                if str(child_key) == key:
                    for leaf_path, tensor in _tensor_leaves(child):
                        output.append(((*child_path, *leaf_path), tensor))
                else:
                    visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(payload)
    output.sort(key=lambda item: item[0])
    return [tensor for _, tensor in output]


def _moment_group_capacities(moment_by_rank: list[list[Any]]) -> tuple[int, ...]:
    if not moment_by_rank:
        return ()
    counts = {len(groups) for groups in moment_by_rank}
    if len(counts) != 1:
        raise ValueError(
            "optimizer shards disagree on moment-group count: "
            f"{sorted(counts)}"
        )
    group_count = next(iter(counts))
    if group_count == 0:
        raise ValueError("optimizer shards contain no Adam moment tensors")
    return tuple(
        sum(int(rank[group_index].numel()) for rank in moment_by_rank)
        for group_index in range(group_count)
    )


def _uses_weight_decay(name: str, shape: tuple[int, ...]) -> bool:
    """Match GPT-NeoX's matrix/embedding versus one-dimensional split."""

    del name
    return len(shape) > 1


def _group_numel(group: Mapping[str, tuple[int, ...]]) -> int:
    return sum(legacy._numel(shape) for shape in group.values())


def _aligned_capacity(expected: int, partition_count: int) -> int:
    if partition_count <= 0:
        raise ValueError(f"partition_count must be positive, got {partition_count}")
    return ((expected + partition_count - 1) // partition_count) * partition_count


def _flatten_shape_groups(
    groups: Sequence[Mapping[str, tuple[int, ...]]],
) -> OrderedDict[str, tuple[int, ...]]:
    complete: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    for group in groups:
        for name, shape in group.items():
            if name in complete:
                raise ValueError(
                    "checkpoint param_shapes contains a duplicate parameter name: "
                    f"{name}"
                )
            complete[name] = tuple(shape)
    return complete


def _direct_capacity_report(
    groups: list[OrderedDict[str, tuple[int, ...]]],
    capacities: tuple[int, ...],
    *,
    partition_count: int,
) -> dict[str, Any] | None:
    if len(groups) != len(capacities):
        return None
    expected = [_group_numel(group) for group in groups]
    aligned = [_aligned_capacity(size, partition_count) for size in expected]
    if aligned != list(capacities):
        return None
    return {
        "method": "checkpoint_group_capacities",
        "coverage": "complete",
        "group_count": len(groups),
        "expected_group_numel": expected,
        "observed_group_numel": list(capacities),
        "padding_numel": [
            observed - raw
            for observed, raw in zip(capacities, expected, strict=True)
        ],
        "parameter_counts": [len(group) for group in groups],
        "partition_count": partition_count,
        "partial_native_state": False,
    }


def _infer_shape_groups(
    shape_groups: list[OrderedDict[str, tuple[int, ...]]],
    capacities: tuple[int, ...],
    *,
    partition_count: int,
) -> tuple[list[OrderedDict[str, tuple[int, ...]]], dict[str, Any] | None]:
    """Reconcile any shape contract with the observed native moment groups.

    The function is pure and reload-safe. It does not mutate another module or
    retain a pointer to a previously installed wrapper.
    """

    direct = _direct_capacity_report(
        shape_groups,
        capacities,
        partition_count=partition_count,
    )
    if direct is not None:
        return shape_groups, None

    complete = _flatten_shape_groups(shape_groups)
    decay = OrderedDict(
        (name, shape)
        for name, shape in complete.items()
        if _uses_weight_decay(name, shape)
    )
    no_decay = OrderedDict(
        (name, shape)
        for name, shape in complete.items()
        if not _uses_weight_decay(name, shape)
    )
    if not complete or not decay or not no_decay:
        raise ValueError(
            "the checkpoint shape contract cannot form both GPT-NeoX "
            "weight-decay and no-decay parameter classes"
        )

    if len(capacities) == 1:
        observed = capacities[0]
        candidates: list[
            tuple[str, OrderedDict[str, tuple[int, ...]], OrderedDict[str, tuple[int, ...]]]
        ] = []
        for coverage, available, unavailable in (
            ("complete", complete, OrderedDict()),
            ("weight_decay_only", decay, no_decay),
            ("no_decay_only", no_decay, decay),
        ):
            if observed == _aligned_capacity(
                _group_numel(available),
                partition_count,
            ):
                candidates.append((coverage, available, unavailable))

        if len(candidates) != 1:
            sizes = {
                "complete": _group_numel(complete),
                "weight_decay_only": _group_numel(decay),
                "no_decay_only": _group_numel(no_decay),
            }
            aligned = {
                name: _aligned_capacity(size, partition_count)
                for name, size in sizes.items()
            }
            qualifier = "does not match" if not candidates else "is ambiguous under"
            raise ValueError(
                "single native Adam moment group "
                f"{qualifier} the GPT-NeoX parameter contract: "
                f"observed={observed}, raw sizes={sizes}, "
                f"aligned capacities={aligned}"
            )

        coverage, available, unavailable = candidates[0]
        if coverage == "complete":
            return [available], None

        expected = _group_numel(available)
        report = {
            "method": "single_native_moment_group_capacity",
            "rule": "rank_gt_1_weight_decay_else_no_decay",
            "alignment_rule": "ceil(numel / partition_count) * partition_count",
            "coverage": coverage,
            "group_count": 1,
            "expected_group_numel": [expected],
            "observed_group_numel": [observed],
            "padding_numel": [observed - expected],
            "parameter_counts": [len(available)],
            "partition_count": partition_count,
            "maximum_allowed_padding_per_group": partition_count - 1,
            "unavailable_parameter_class": (
                "one_dimensional_no_decay"
                if coverage == "weight_decay_only"
                else "matrix_and_embedding_weight_decay"
            ),
            "unavailable_parameter_count": len(unavailable),
            "unavailable_parameter_numel": _group_numel(unavailable),
            "partial_native_state": True,
        }
        return [available], report

    if len(capacities) == 2:
        candidates: list[
            tuple[
                str,
                list[OrderedDict[str, tuple[int, ...]]],
                list[int],
            ]
        ] = []
        for order, groups in (
            ("weight_decay_then_no_decay", [decay, no_decay]),
            ("no_decay_then_weight_decay", [no_decay, decay]),
        ):
            expected = [_group_numel(group) for group in groups]
            aligned = [
                _aligned_capacity(size, partition_count)
                for size in expected
            ]
            if aligned == list(capacities):
                candidates.append((order, groups, expected))

        if len(candidates) != 1:
            candidate_sizes = {
                "weight_decay_then_no_decay": [
                    _group_numel(decay),
                    _group_numel(no_decay),
                ],
                "no_decay_then_weight_decay": [
                    _group_numel(no_decay),
                    _group_numel(decay),
                ],
            }
            raise ValueError(
                "native Adam groups do not identify a unique GPT-NeoX "
                f"decay/no-decay ordering: observed={list(capacities)}, "
                f"candidate sizes={candidate_sizes}"
            )

        order, groups, expected = candidates[0]
        report = {
            "method": "optimizer_moment_group_capacities",
            "rule": "rank_gt_1_weight_decay_else_no_decay",
            "order": order,
            "coverage": "complete",
            "group_count": 2,
            "expected_group_numel": expected,
            "observed_group_numel": list(capacities),
            "padding_numel": [
                observed - raw
                for observed, raw in zip(capacities, expected, strict=True)
            ],
            "parameter_counts": [len(group) for group in groups],
            "partition_count": partition_count,
            "partial_native_state": False,
        }
        return groups, report

    raise ValueError(
        "checkpoint shape groups do not match the native Adam group count: "
        f"shape_groups={len(shape_groups)}, moment_groups={len(capacities)}, "
        f"capacities={list(capacities)}"
    )


def _master_group_count(
    optimizer_payloads: list[Mapping[str, Any]],
) -> int | None:
    extracted = [legacy._flat_partition_groups(payload) for payload in optimizer_payloads]
    concrete = [item for item in extracted if item is not None]
    if not concrete:
        return None
    counts = {len(item[1]) for item in concrete}
    if len(counts) != 1:
        raise ValueError(
            "optimizer shards disagree on fp32 master group count: "
            f"{sorted(counts)}"
        )
    return next(iter(counts))


def reconstruct_zero_adam_state(
    checkpoint_dir: str | Path,
    *,
    parameter_names: Iterable[str] | None = None,
    parameter_shapes: Mapping[str, Sequence[int]]
    | Sequence[Mapping[str, Sequence[int]]]
    | None = None,
    reference_signatures: Mapping[str, Mapping[str, float]] | None = None,
    signature_rtol: float = 0.02,
    minimum_match_fraction: float = 0.95,
) -> legacy.NativeAdamState:
    """Reconstruct native ZeRO Adam moments without import-time patching."""

    import torch

    root = Path(checkpoint_dir)
    optimizer_files = tuple(
        sorted(
            root.glob("zero_pp_rank_*_optim_states.pt"),
            key=lambda path: legacy._rank_key(path.name),
        )
    )
    if not optimizer_files:
        raise FileNotFoundError(f"no ZeRO optimizer shards found under {root}")

    optimizer_payloads = [legacy.load_torch_payload(path) for path in optimizer_files]
    model_files = tuple(sorted(root.glob("*model_states.pt")))
    model_payloads = [legacy.load_torch_payload(path) for path in model_files]

    avg_by_rank = [_moment_tensors(payload, "exp_avg") for payload in optimizer_payloads]
    sq_by_rank = [_moment_tensors(payload, "exp_avg_sq") for payload in optimizer_payloads]
    avg_capacities = _moment_group_capacities(avg_by_rank)
    sq_capacities = _moment_group_capacities(sq_by_rank)
    if avg_capacities != sq_capacities:
        raise ValueError(
            "first- and second-moment group capacities disagree: "
            f"exp_avg={list(avg_capacities)}, "
            f"exp_avg_sq={list(sq_capacities)}"
        )

    param_groups = legacy._find_param_shapes(model_payloads)
    shape_source = "model_checkpoint_param_shapes"
    validation: dict[str, Any] = {
        "method": "checkpoint_metadata",
        "validated": True,
    }

    if param_groups is None:
        param_groups = legacy._find_param_shapes(optimizer_payloads)
        shape_source = "optimizer_checkpoint_param_shapes"

    if param_groups is not None:
        param_groups, metadata_report = _infer_shape_groups(
            param_groups,
            avg_capacities,
            partition_count=len(optimizer_files),
        )
        if metadata_report is not None:
            shape_source = f"{shape_source}_reconciled_with_native_moment_capacity"
            validation = {
                "method": "checkpoint_metadata_plus_optimizer_group_capacities",
                "validated": True,
                "optimizer_groups": metadata_report,
            }

    if param_groups is None:
        caller_groups = legacy._caller_shape_groups(parameter_shapes)
        if caller_groups is None:
            model_keys = sorted(
                {
                    str(key)
                    for payload in model_payloads
                    for key in payload.keys()
                }
            )
            optimizer_keys = sorted(
                {
                    str(key)
                    for payload in optimizer_payloads[:1]
                    for key in payload.keys()
                }
            )
            raise KeyError(
                "could not find param_shapes in native checkpoint metadata. "
                "Supply the loaded model's ordered parameter_shapes and "
                "reference_signatures for validated reconstruction. "
                f"model top-level keys={model_keys}; "
                f"optimizer top-level keys={optimizer_keys}"
            )

        param_groups, group_report = _infer_shape_groups(
            caller_groups,
            avg_capacities,
            partition_count=len(optimizer_files),
        )
        shape_source = (
            "validated_model_parameter_order_with_inferred_optimizer_groups"
            if group_report is not None
            else "validated_model_parameter_order"
        )

        try:
            validation = legacy._validate_caller_order(
                optimizer_payloads,
                param_groups,
                reference_signatures,
                signature_rtol=signature_rtol,
                minimum_match_fraction=minimum_match_fraction,
            )
        except ValueError as grouped_error:
            master_groups = _master_group_count(optimizer_payloads)
            if group_report is None or master_groups != 1:
                raise
            global_validation = legacy._validate_caller_order(
                optimizer_payloads,
                caller_groups,
                reference_signatures,
                signature_rtol=signature_rtol,
                minimum_match_fraction=minimum_match_fraction,
            )
            validation = {
                "method": "global_fp32_signatures_plus_optimizer_group_capacities",
                "validated": True,
                "match_fraction": global_validation["match_fraction"],
                "global_order_validation": global_validation,
                "group_specific_master_validation_unavailable": str(grouped_error),
            }

        validation["validated"] = True
        if group_report is not None:
            validation["optimizer_groups"] = group_report

    group_count = len(param_groups)
    if any(len(groups) < group_count for groups in avg_by_rank + sq_by_rank):
        observed = [
            (len(avg), len(sq))
            for avg, sq in zip(avg_by_rank, sq_by_rank, strict=True)
        ]
        raise ValueError(
            "optimizer shards expose fewer moment groups than the shape "
            f"contract: observed {observed}, expected at least {group_count}"
        )

    requested = None if parameter_names is None else set(parameter_names)
    exp_avg: dict[str, Any] = {}
    exp_avg_sq: dict[str, Any] = {}
    total_parameters = 0

    for group_index, shapes in enumerate(param_groups):
        expected = _group_numel(shapes)
        flat_avg = torch.cat(
            [rank[group_index] for rank in avg_by_rank]
        )[:expected]
        flat_sq = torch.cat(
            [rank[group_index] for rank in sq_by_rank]
        )[:expected]
        if flat_avg.numel() != expected or flat_sq.numel() != expected:
            raise ValueError(
                f"group {group_index} reconstruction incomplete: expected "
                f"{expected}, got {flat_avg.numel()} and {flat_sq.numel()}"
            )

        offset = 0
        for name, shape in shapes.items():
            count = legacy._numel(shape)
            if requested is None or name in requested:
                exp_avg[name] = (
                    flat_avg[offset : offset + count]
                    .reshape(shape)
                    .clone()
                )
                exp_avg_sq[name] = (
                    flat_sq[offset : offset + count]
                    .reshape(shape)
                    .clone()
                )
            offset += count
            total_parameters += count

    if requested is not None:
        missing = requested - set(exp_avg)
        if missing:
            raise KeyError(
                "requested parameters absent from native state: "
                f"{sorted(missing)}"
            )

    return legacy.NativeAdamState(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        source_files=tuple(str(path) for path in optimizer_files),
        partition_count=len(optimizer_files),
        parameter_count=total_parameters,
        shape_source=shape_source,
        shape_validation=validation,
    )


__all__ = ["reconstruct_zero_adam_state"]
