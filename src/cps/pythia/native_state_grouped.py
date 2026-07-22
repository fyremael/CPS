from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import native_state as legacy


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
    return tuple(
        sum(int(rank[group_index].numel()) for rank in moment_by_rank)
        for group_index in range(group_count)
    )


def _uses_weight_decay(name: str, shape: tuple[int, ...]) -> bool:
    """Return the GPT-NeoX/Megatron weight-decay classification.

    Matrix and embedding parameters receive weight decay. Biases, LayerNorm
    scales, and other one-dimensional parameters are assigned to the no-decay
    group. This is the split used by the released Pythia-70M optimizer state.
    """

    del name
    return len(shape) > 1


def _capacity_report(
    groups: list[OrderedDict[str, tuple[int, ...]]],
    capacities: tuple[int, ...],
    *,
    partition_count: int,
    order: str,
) -> dict[str, Any] | None:
    if len(groups) != len(capacities):
        return None

    expected = tuple(
        sum(legacy._numel(shape) for shape in group.values())
        for group in groups
    )
    padding = tuple(
        capacity - wanted
        for capacity, wanted in zip(capacities, expected, strict=True)
    )

    # DeepSpeed ZeRO-2 aligns flattened groups to 2 * data-parallel world size.
    maximum_padding = 2 * partition_count
    if any(value < 0 or value > maximum_padding for value in padding):
        return None

    return {
        "method": "optimizer_moment_group_capacities",
        "rule": "rank_gt_1_weight_decay_else_no_decay",
        "order": order,
        "group_count": len(groups),
        "expected_group_numel": list(expected),
        "observed_group_numel": list(capacities),
        "padding_numel": list(padding),
        "parameter_counts": [len(group) for group in groups],
        "maximum_allowed_padding_per_group": maximum_padding,
    }


def _infer_optimizer_shape_groups(
    caller_groups: list[OrderedDict[str, tuple[int, ...]]],
    capacities: tuple[int, ...],
    *,
    partition_count: int,
) -> tuple[list[OrderedDict[str, tuple[int, ...]]], dict[str, Any] | None]:
    """Infer the two GPT-NeoX optimizer groups from an ordered model contract.

    The fallback is accepted only when exactly one decay/no-decay ordering fits
    the observed flattened moment capacities, allowing only bounded DeepSpeed
    partition padding.
    """

    if len(caller_groups) != 1 or len(capacities) <= 1:
        return caller_groups, None
    if len(capacities) != 2:
        raise ValueError(
            "native checkpoint omits param_shapes and exposes "
            f"{len(capacities)} optimizer moment groups. CPS has a governed "
            "fallback only for the two-group GPT-NeoX decay/no-decay contract."
        )

    complete = caller_groups[0]
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
    if not decay or not no_decay:
        raise ValueError(
            "could not form both GPT-NeoX weight-decay and no-decay groups "
            "from the loaded model contract"
        )

    candidates = (
        ("weight_decay_then_no_decay", [decay, no_decay]),
        ("no_decay_then_weight_decay", [no_decay, decay]),
    )
    feasible: list[
        tuple[list[OrderedDict[str, tuple[int, ...]]], dict[str, Any]]
    ] = []
    for order, groups in candidates:
        report = _capacity_report(
            groups,
            capacities,
            partition_count=partition_count,
            order=order,
        )
        if report is not None:
            feasible.append((groups, report))

    if len(feasible) == 1:
        return feasible[0]

    candidate_sizes = {
        order: [
            sum(legacy._numel(shape) for shape in group.values())
            for group in groups
        ]
        for order, groups in candidates
    }
    if not feasible:
        raise ValueError(
            "caller model contract does not fit the native optimizer moment "
            f"groups: observed capacities={list(capacities)}, "
            f"candidate sizes={candidate_sizes}. This is not permissible "
            "DeepSpeed partition padding."
        )
    raise ValueError(
        "native optimizer-group order is ambiguous under the capacity contract: "
        f"observed capacities={list(capacities)}, candidate sizes={candidate_sizes}"
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
    """Reconstruct ZeRO Adam moments with governed optimizer-group recovery.

    Checkpoint-authored ``param_shapes`` remain authoritative. When historical
    metadata omits that field, CPS validates the Transformers parameter order
    against native fp32 master weights and infers GPT-NeoX's separate decay and
    no-decay groups from their exact native moment capacities.
    """

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

    avg_by_rank = [
        legacy._moment_tensors(payload, "exp_avg")
        for payload in optimizer_payloads
    ]
    sq_by_rank = [
        legacy._moment_tensors(payload, "exp_avg_sq")
        for payload in optimizer_payloads
    ]
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

    # A historical packet can publish a complete one-group param_shapes mapping
    # while exposing only a filtered Adam moment group. Reconcile every
    # checkpoint-authored shape contract against the observed moment capacities,
    # not only the caller-supplied fallback contract. The packet compatibility
    # layer installs the single-group decay/no-decay classifier used here.
    if param_groups is not None:
        param_groups, metadata_group_report = _infer_optimizer_shape_groups(
            param_groups,
            avg_capacities,
            partition_count=len(optimizer_files),
        )
        if metadata_group_report is not None:
            shape_source = f"{shape_source}_reconciled_with_native_moment_capacity"
            validation = {
                "method": "checkpoint_metadata_plus_optimizer_group_capacities",
                "validated": True,
                "optimizer_groups": metadata_group_report,
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

        param_groups, group_report = _infer_optimizer_shape_groups(
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
            # Some legacy packets expose one combined fp32 master group while
            # retaining two Adam moment groups. In that case validate the global
            # order against the master weights and the group split by exact
            # native moment capacities. Any other mismatch remains fatal.
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
        expected = sum(legacy._numel(shape) for shape in shapes.values())
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
