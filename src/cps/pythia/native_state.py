from __future__ import annotations

import math
import sys
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class NativeAdamState:
    exp_avg: dict[str, Any]
    exp_avg_sq: dict[str, Any]
    source_files: tuple[str, ...]
    partition_count: int
    parameter_count: int
    shape_source: str = "checkpoint_param_shapes"
    shape_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeCheckpointSummary:
    optimizer_shards: tuple[str, ...]
    model_state_files: tuple[str, ...]
    yaml_files: tuple[str, ...]
    optimizer_payload_keys: tuple[str, ...]
    model_payload_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def install_deepspeed_pickle_shims() -> None:
    """Install minimal modules needed to unpickle historical Pythia checkpoints.

    The optimizer tensors are the target. Historical loss-scaler objects are not
    executed and are represented by inert placeholders when DeepSpeed is absent.
    """

    try:
        import deepspeed.runtime.fp16.loss_scaler  # type: ignore[import-not-found]  # noqa: F401
        return
    except Exception:
        pass

    modules = {
        "deepspeed": types.ModuleType("deepspeed"),
        "deepspeed.runtime": types.ModuleType("deepspeed.runtime"),
        "deepspeed.runtime.fp16": types.ModuleType("deepspeed.runtime.fp16"),
        "deepspeed.runtime.fp16.loss_scaler": types.ModuleType(
            "deepspeed.runtime.fp16.loss_scaler"
        ),
    }

    class DynamicLossScaler:  # pragma: no cover - used only while unpickling external files
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    modules["deepspeed.runtime.fp16.loss_scaler"].DynamicLossScaler = DynamicLossScaler
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def load_torch_payload(path: str | Path) -> Mapping[str, Any]:
    import torch

    install_deepspeed_pickle_shims()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload at {path} is not a mapping")
    return payload


def discover_native_checkpoint(path: str | Path) -> NativeCheckpointSummary:
    root = Path(path)
    optimizer = tuple(sorted(str(p) for p in root.glob("zero_pp_rank_*_optim_states.pt")))
    model = tuple(sorted(str(p) for p in root.glob("*model_states.pt")))
    yamls = tuple(sorted(str(p) for p in root.glob("*.yml")))
    optimizer_keys: tuple[str, ...] = ()
    model_keys: tuple[str, ...] = ()
    if optimizer:
        optimizer_keys = tuple(sorted(load_torch_payload(optimizer[0]).keys()))
    if model:
        model_keys = tuple(sorted(load_torch_payload(model[0]).keys()))
    return NativeCheckpointSummary(
        optimizer_shards=optimizer,
        model_state_files=model,
        yaml_files=yamls,
        optimizer_payload_keys=optimizer_keys,
        model_payload_keys=model_keys,
    )


def _moment_tensors(payload: Mapping[str, Any], key: str) -> list[Any]:
    import torch

    output: list[tuple[tuple[str, ...], Any]] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            if key in value and torch.is_tensor(value[key]):
                output.append(((*path, key), value[key].detach().cpu().reshape(-1)))
            for child_key, child in value.items():
                visit(child, (*path, str(child_key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(payload)
    output.sort(key=lambda item: item[0])
    return [tensor for _, tensor in output]


def _coerce_param_shapes(value: Any) -> list[OrderedDict[str, tuple[int, ...]]] | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    groups: list[OrderedDict[str, tuple[int, ...]]] = []
    for group in value:
        if not isinstance(group, Mapping):
            continue
        converted: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        for name, shape in group.items():
            try:
                converted[str(name)] = tuple(int(x) for x in shape)
            except TypeError:
                return None
        if converted:
            groups.append(converted)
    return groups or None


def _find_param_shapes(
    payloads: Iterable[Mapping[str, Any]],
) -> list[OrderedDict[str, tuple[int, ...]]] | None:
    def search(value: Any) -> Any:
        if isinstance(value, Mapping):
            if "param_shapes" in value:
                return value["param_shapes"]
            for child in value.values():
                found = search(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = search(child)
                if found is not None:
                    return found
        return None

    for payload in payloads:
        groups = _coerce_param_shapes(search(payload))
        if groups:
            return groups
    return None


def _flat_partition_groups(payload: Mapping[str, Any]) -> tuple[str, list[Any]] | None:
    """Find fp32 master-weight partitions stored by ZeRO-1/2/3 variants."""

    import torch

    candidates = (
        "single_partition_of_fp32_groups",
        "fp32_flat_groups",
        "single_partition_of_fp32_groups_flat",
    )

    def search(value: Any) -> tuple[str, list[Any]] | None:
        if isinstance(value, Mapping):
            for key in candidates:
                candidate = value.get(key)
                if torch.is_tensor(candidate):
                    return key, [candidate.detach().cpu().reshape(-1)]
                if isinstance(candidate, (list, tuple)) and candidate and all(
                    torch.is_tensor(item) for item in candidate
                ):
                    return key, [item.detach().cpu().reshape(-1) for item in candidate]
            for child in value.values():
                found = search(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = search(child)
                if found is not None:
                    return found
        return None

    return search(payload)


def _caller_shape_groups(
    parameter_shapes: Mapping[str, Sequence[int]] | Sequence[Mapping[str, Sequence[int]]] | None,
) -> list[OrderedDict[str, tuple[int, ...]]] | None:
    if parameter_shapes is None:
        return None
    return _coerce_param_shapes(parameter_shapes)


def _relative_error(observed: float, expected: float) -> float:
    scale = max(abs(expected), 1.0e-12)
    return abs(observed - expected) / scale


def _validate_caller_order(
    optimizer_payloads: list[Mapping[str, Any]],
    param_groups: list[OrderedDict[str, tuple[int, ...]]],
    reference_signatures: Mapping[str, Mapping[str, float]] | None,
    *,
    signature_rtol: float,
    minimum_match_fraction: float,
) -> dict[str, Any]:
    """Validate caller-supplied parameter order against native fp32 master weights.

    Some historical GPT-NeoX pipeline metadata omits ``param_shapes``. In that
    case the Transformers model supplies candidate names and shapes. CPS accepts
    that order only when tensorwise L2 and L1 signatures of the reconstructed
    fp32 master weights agree with the loaded checkpoint. Permutations and
    transposes used during format conversion preserve these signatures.
    """

    import torch

    if reference_signatures is None:
        raise KeyError(
            "native checkpoint metadata does not contain param_shapes, and no model-reference "
            "signatures were supplied to validate a caller-provided parameter order"
        )

    extracted = [_flat_partition_groups(payload) for payload in optimizer_payloads]
    if any(item is None for item in extracted):
        raise KeyError(
            "native checkpoint metadata does not contain param_shapes and the optimizer shards "
            "do not expose fp32 master-weight partitions needed to validate model-order fallback"
        )
    concrete = [item for item in extracted if item is not None]
    keys = {item[0] for item in concrete}
    if len(keys) != 1:
        raise ValueError(f"optimizer shards disagree on fp32 master-weight field: {sorted(keys)}")
    key = next(iter(keys))
    partitions = [item[1] for item in concrete]
    group_count = len(param_groups)
    if any(len(groups) < group_count for groups in partitions):
        observed = [len(groups) for groups in partitions]
        raise ValueError(
            "optimizer shards expose fewer fp32 groups than the caller shape contract: "
            f"observed {observed}, expected {group_count}"
        )

    checked = 0
    matched = 0
    worst_error = 0.0
    mismatches: list[dict[str, Any]] = []
    for group_index, shapes in enumerate(param_groups):
        expected_numel = sum(_numel(shape) for shape in shapes.values())
        flat = torch.cat([rank[group_index] for rank in partitions])[:expected_numel]
        if flat.numel() != expected_numel:
            raise ValueError(
                f"fp32 master group {group_index} is incomplete: expected {expected_numel}, "
                f"got {flat.numel()}"
            )
        offset = 0
        for name, shape in shapes.items():
            count = _numel(shape)
            segment = flat[offset : offset + count]
            offset += count
            reference = reference_signatures.get(name)
            if reference is None:
                continue
            observed_l2 = float(torch.linalg.vector_norm(segment.float()))
            observed_l1 = float(segment.float().abs().sum())
            l2_error = _relative_error(observed_l2, float(reference["l2"]))
            l1_error = _relative_error(observed_l1, float(reference["l1"]))
            error = max(l2_error, l1_error)
            worst_error = max(worst_error, error)
            checked += 1
            if error <= signature_rtol:
                matched += 1
            elif len(mismatches) < 8:
                mismatches.append(
                    {
                        "name": name,
                        "l2_relative_error": l2_error,
                        "l1_relative_error": l1_error,
                    }
                )

    if checked == 0:
        raise ValueError("no caller parameter signatures could be matched to the shape contract")
    match_fraction = matched / checked
    result = {
        "method": "fp32_master_tensor_signatures",
        "master_weight_field": key,
        "checked_tensors": checked,
        "matched_tensors": matched,
        "match_fraction": match_fraction,
        "minimum_match_fraction": minimum_match_fraction,
        "signature_rtol": signature_rtol,
        "worst_relative_error": worst_error,
        "mismatches": mismatches,
    }
    if match_fraction < minimum_match_fraction:
        raise ValueError(
            "caller-supplied parameter order failed validation against native fp32 master "
            f"weights: {matched}/{checked} tensor signatures matched at rtol={signature_rtol}; "
            f"first mismatches={mismatches}"
        )
    return result


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
) -> NativeAdamState:
    """Offline reconstruction of ZeRO-partitioned Adam moments.

    The preferred shape contract is the checkpoint's own ``param_shapes`` field.
    Some historical GPT-NeoX pipeline checkpoints omit that field. For those
    packets, a caller may provide the loaded model's ordered parameter shapes and
    compact tensor signatures. CPS then validates that ordering against the native
    fp32 master-weight partitions before assigning any Adam moments to names.
    """

    import torch

    root = Path(checkpoint_dir)
    optimizer_files = tuple(
        sorted(root.glob("zero_pp_rank_*_optim_states.pt"), key=lambda p: _rank_key(p.name))
    )
    if not optimizer_files:
        raise FileNotFoundError(f"no ZeRO optimizer shards found under {root}")
    optimizer_payloads = [load_torch_payload(path) for path in optimizer_files]
    model_files = tuple(sorted(root.glob("*model_states.pt")))
    model_payloads = [load_torch_payload(path) for path in model_files]

    param_groups = _find_param_shapes(model_payloads)
    shape_source = "model_checkpoint_param_shapes"
    validation: dict[str, Any] = {"method": "checkpoint_metadata", "validated": True}
    if param_groups is None:
        param_groups = _find_param_shapes(optimizer_payloads)
        shape_source = "optimizer_checkpoint_param_shapes"
    if param_groups is None:
        param_groups = _caller_shape_groups(parameter_shapes)
        shape_source = "validated_model_parameter_order"
        if param_groups is None:
            model_keys = sorted({str(key) for payload in model_payloads for key in payload.keys()})
            optimizer_keys = sorted(
                {str(key) for payload in optimizer_payloads[:1] for key in payload.keys()}
            )
            raise KeyError(
                "could not find param_shapes in native checkpoint metadata. "
                "Supply the loaded model's ordered parameter_shapes and reference_signatures "
                "for validated reconstruction. "
                f"model top-level keys={model_keys}; optimizer top-level keys={optimizer_keys}"
            )
        validation = _validate_caller_order(
            optimizer_payloads,
            param_groups,
            reference_signatures,
            signature_rtol=signature_rtol,
            minimum_match_fraction=minimum_match_fraction,
        )
        validation["validated"] = True

    avg_by_rank = [_moment_tensors(payload, "exp_avg") for payload in optimizer_payloads]
    sq_by_rank = [_moment_tensors(payload, "exp_avg_sq") for payload in optimizer_payloads]
    group_count = len(param_groups)
    if any(len(groups) < group_count for groups in avg_by_rank + sq_by_rank):
        observed = [(len(a), len(b)) for a, b in zip(avg_by_rank, sq_by_rank, strict=True)]
        raise ValueError(
            f"optimizer shards expose fewer moment groups than the shape contract; "
            f"observed {observed}, expected at least {group_count}"
        )

    requested = None if parameter_names is None else set(parameter_names)
    exp_avg: dict[str, Any] = {}
    exp_avg_sq: dict[str, Any] = {}
    total_parameters = 0
    for group_index, shapes in enumerate(param_groups):
        expected = sum(_numel(shape) for shape in shapes.values())
        flat_avg = torch.cat([rank[group_index] for rank in avg_by_rank])[:expected]
        flat_sq = torch.cat([rank[group_index] for rank in sq_by_rank])[:expected]
        if flat_avg.numel() != expected or flat_sq.numel() != expected:
            raise ValueError(
                f"group {group_index} reconstruction incomplete: expected {expected}, "
                f"got {flat_avg.numel()} and {flat_sq.numel()}"
            )
        offset = 0
        for name, shape in shapes.items():
            count = _numel(shape)
            if requested is None or name in requested:
                exp_avg[name] = flat_avg[offset : offset + count].reshape(shape).clone()
                exp_avg_sq[name] = flat_sq[offset : offset + count].reshape(shape).clone()
            offset += count
            total_parameters += count

    if requested is not None:
        missing = requested - set(exp_avg)
        if missing:
            raise KeyError(f"requested parameters absent from native state: {sorted(missing)}")
    return NativeAdamState(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        source_files=tuple(str(path) for path in optimizer_files),
        partition_count=len(optimizer_files),
        parameter_count=total_parameters,
        shape_source=shape_source,
        shape_validation=validation,
    )


def parameter_reference_signatures(parameters: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Compute compact permutation-invariant signatures for checkpoint-order validation."""

    import torch

    output: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = parameter.detach().float()
            output[name] = {
                "l2": float(torch.linalg.vector_norm(value).cpu()),
                "l1": float(value.abs().sum().cpu()),
            }
    return output


def _rank_key(name: str) -> tuple[int, str]:
    import re

    match = re.search(r"zero_pp_rank_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)
