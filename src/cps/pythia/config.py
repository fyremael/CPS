from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    run: str = "pythia-70m"
    model_id: str | None = None
    native_checkpoint_id: str | None = None
    revision: str = "step0"
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    cache_dir: str = ".cache/huggingface"


@dataclass(frozen=True)
class DataConfig:
    source: str = "builtin"
    sequence_length: int = 64
    batch_size: int = 1
    seed: int = 17
    token_file: str | None = None
    token_dtype: str = "uint16"
    start_index: int = 0
    prompts: tuple[str, ...] = (
        "Mathematics is the study of structure, change, and relation.",
        "A stable optimizer should control both asymptotic and transient growth.",
    )


@dataclass(frozen=True)
class StateConfig:
    moment_source: str = "reconstructed"
    native_checkpoint_dir: str | None = None
    native_download: bool = False
    step: int = 1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    v_floor: float = 1e-12


@dataclass(frozen=True)
class BasisConfig:
    parameter_patterns: tuple[str, ...] = (
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.dense.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    )
    max_selected_numel: int = 8_000_000
    components: tuple[str, ...] = ("theta", "momentum", "update", "random")
    random_vectors_per_block: int = 1
    rank: int = 16
    seed: int = 23
    orthogonalization_tolerance: float = 1e-7


@dataclass(frozen=True)
class JacobianConfig:
    mode: str = "autodiff"
    finite_difference_relative_step: float = 1e-4
    finite_difference_absolute_floor: float = 1e-6
    closure_residual: bool = True


@dataclass(frozen=True)
class SweepConfig:
    phase_count: int = 33
    finite_horizon: int = 16
    compute_kreiss: bool = False
    coupling_selection: str = "largest_offdiagonal"
    maximum_couplings: int = 16
    block_channel: int = 0


@dataclass(frozen=True)
class OutputConfig:
    root: str = "artifacts/pythia"
    run_name: str = "probe"
    save_reduced_operator: bool = True
    save_trajectories: bool = True


@dataclass(frozen=True)
class PythiaProbeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    state: StateConfig = field(default_factory=StateConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    jacobian: JacobianConfig = field(default_factory=JacobianConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    if not raw:
        return cls()
    fields = cls.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in fields:
            raise ValueError(f"unknown {cls.__name__} field: {name}")
        default = fields[name].default
        if isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_probe_config(path: str | Path) -> PythiaProbeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"model", "data", "state", "basis", "jacobian", "sweep", "output"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    return PythiaProbeConfig(
        model=_construct(ModelConfig, payload.get("model")),
        data=_construct(DataConfig, payload.get("data")),
        state=_construct(StateConfig, payload.get("state")),
        basis=_construct(BasisConfig, payload.get("basis")),
        jacobian=_construct(JacobianConfig, payload.get("jacobian")),
        sweep=_construct(SweepConfig, payload.get("sweep")),
        output=_construct(OutputConfig, payload.get("output")),
    )
