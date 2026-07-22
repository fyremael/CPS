from __future__ import annotations

from dataclasses import asdict, dataclass


EARLY_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DENSE_STEPS = tuple(range(1000, 143001, 1000))
PYTHIA_STEPS = EARLY_STEPS + DENSE_STEPS


@dataclass(frozen=True)
class PythiaRunSpec:
    key: str
    model_id: str
    native_checkpoint_id: str | None
    parameter_count: int
    family: str
    seed: int | None = None
    data_seed: int | None = None
    weight_seed: int | None = None
    notes: str = ""

    def revision(self, step: int) -> str:
        if step not in PYTHIA_STEPS:
            raise ValueError(f"unsupported Pythia checkpoint step: {step}")
        return f"step{step}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_BASE = {
    "pythia-14m": (14_000_000, None),
    "pythia-31m": (31_000_000, None),
    "pythia-70m": (70_426_624, "EleutherAI/neox-ckpt-pythia-70m"),
    "pythia-160m": (162_322_944, "EleutherAI/neox-ckpt-pythia-160m"),
    "pythia-410m": (405_334_016, "EleutherAI/neox-ckpt-pythia-410m"),
    "pythia-1b": (1_011_781_632, None),
    "pythia-1.4b": (1_414_647_808, None),
    "pythia-2.8b": (2_775_208_960, "EleutherAI/neox-ckpt-pythia-2.8b"),
    "pythia-6.9b": (6_857_302_016, None),
    "pythia-12b": (11_846_072_320, None),
}

_REGISTRY: dict[str, PythiaRunSpec] = {}
for name, (count, native) in _BASE.items():
    _REGISTRY[name] = PythiaRunSpec(
        key=name,
        model_id=f"EleutherAI/{name}",
        native_checkpoint_id=native,
        parameter_count=count,
        family="pythia",
    )

for seed in range(1, 10):
    key = f"polypythia-70m-seed{seed}"
    _REGISTRY[key] = PythiaRunSpec(
        key=key,
        model_id=f"EleutherAI/pythia-70m-seed{seed}",
        native_checkpoint_id=f"EleutherAI/neox-ckpt-pythia-70m-seed{seed}",
        parameter_count=70_426_624,
        family="polypythia",
        seed=seed,
    )

# The exact public repository spellings of the 160M causal variants can be
# overridden in YAML. These logical keys keep experiment manifests stable.
for weight_seed, data_seed in ((1, 1), (1, 2), (2, 1), (2, 2)):
    key = f"polypythia-160m-w{weight_seed}-d{data_seed}"
    _REGISTRY[key] = PythiaRunSpec(
        key=key,
        model_id=f"EleutherAI/pythia-160m-seed{weight_seed}",
        native_checkpoint_id=None,
        parameter_count=162_322_944,
        family="polypythia-causal",
        weight_seed=weight_seed,
        data_seed=data_seed,
        notes="Logical causal-control key; set model_id override in the run config.",
    )


def get_run_spec(key: str) -> PythiaRunSpec:
    try:
        return _REGISTRY[key.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown Pythia run '{key}'. Available: {available}") from exc


def list_run_specs() -> tuple[PythiaRunSpec, ...]:
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))
