from __future__ import annotations

import sys
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NativeAdamState:
    exp_avg: dict[str, Any]
    exp_avg_sq: dict[str, Any]
    source_files: tuple[str, ...]
    partition_count: int
    parameter_count: int


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


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))
    else:
        yield path, value


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


def _find_param_shapes(payloads: Iterable[Mapping[str, Any]]) -> list[OrderedDict[str, tuple[int, ...]]]:
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
        value = search(payload)
        if isinstance(value, (list, tuple)):
            groups: list[OrderedDict[str, tuple[int, ...]]] = []
            for group in value:
                if isinstance(group, Mapping):
                    groups.append(
                        OrderedDict(
                            (str(name), tuple(int(x) for x in shape))
                            for name, shape in group.items()
                        )
                    )
            if groups:
                return groups
    raise KeyError("could not find param_shapes in model checkpoint metadata")


def reconstruct_zero_adam_state(
    checkpoint_dir: str | Path,
    *,
    parameter_names: Iterable[str] | None = None,
) -> NativeAdamState:
    """Offline reconstruction of ZeRO-partitioned Adam moments.

    The routine discovers flat ``exp_avg`` and ``exp_avg_sq`` tensors in each
    data-parallel shard, concatenates rank partitions group-by-group, trims
    padding according to ``param_shapes``, and splits the result by parameter.
    It validates every recovered shape before returning.
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

    avg_by_rank = [_moment_tensors(payload, "exp_avg") for payload in optimizer_payloads]
    sq_by_rank = [_moment_tensors(payload, "exp_avg_sq") for payload in optimizer_payloads]
    group_count = len(param_groups)
    if any(len(groups) < group_count for groups in avg_by_rank + sq_by_rank):
        observed = [(len(a), len(b)) for a, b in zip(avg_by_rank, sq_by_rank, strict=True)]
        raise ValueError(
            f"optimizer shards expose fewer moment groups than param_shapes; observed {observed}, "
            f"expected at least {group_count}"
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
    )


def _rank_key(name: str) -> tuple[int, str]:
    import re

    match = re.search(r"zero_pp_rank_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result
