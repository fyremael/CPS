from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from .analysis import analyze_reduced_operator, save_analysis
from .basis import build_semantic_basis
from .blocks import select_parameter_names
from .config import PythiaProbeConfig
from .data import build_batch
from .functional_adamw import AdamWHyperparameters, FunctionalAdamWProbe
from .native_state import reconstruct_zero_adam_state
from .reduced_operator import project_jacobian
from .registry import get_run_spec


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def run_probe(config: PythiaProbeConfig) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra before running a Pythia probe") from exc

    started = time.time()
    spec = get_run_spec(config.model.run)
    model_id = config.model.model_id or spec.model_id
    revision = config.model.revision
    device = _device(config.model.device)
    dtype = _dtype(config.model.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=config.model.cache_dir,
        torch_dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    ).to(device)
    model.eval()

    all_named = dict(model.named_parameters())
    selected_names = select_parameter_names(all_named, config.basis.parameter_patterns)
    if not selected_names:
        raise RuntimeError("no parameters matched basis.parameter_patterns")
    selected_numel = sum(all_named[name].numel() for name in selected_names)
    if selected_numel > config.basis.max_selected_numel:
        raise RuntimeError(
            f"selected parameter count {selected_numel:,} exceeds max_selected_numel "
            f"{config.basis.max_selected_numel:,}"
        )

    batch = build_batch(tokenizer, config.data, device)
    initial_m = None
    initial_v = None
    native_metadata: dict[str, Any] = {"mode": config.state.moment_source}
    if config.state.moment_source == "native":
        if config.state.native_checkpoint_dir is None:
            raise ValueError("native_checkpoint_dir is required for moment_source=native")
        native = reconstruct_zero_adam_state(
            config.state.native_checkpoint_dir,
            parameter_names=selected_names,
        )
        initial_m = _align_native_names(native.exp_avg, selected_names)
        initial_v = _align_native_names(native.exp_avg_sq, selected_names)
        native_metadata.update(
            {
                "partition_count": native.partition_count,
                "source_files": list(native.source_files),
                "parameter_count": native.parameter_count,
            }
        )
    elif config.state.moment_source not in {"reconstructed", "zero"}:
        raise ValueError(f"unsupported moment_source: {config.state.moment_source}")

    hyper = AdamWHyperparameters(
        learning_rate=config.state.learning_rate,
        beta1=config.state.beta1,
        beta2=config.state.beta2,
        epsilon=config.state.epsilon,
        weight_decay=config.state.weight_decay,
        step=config.state.step,
        v_floor=config.state.v_floor,
    )
    probe = FunctionalAdamWProbe(
        model,
        batch,
        selected_names,
        hyper,
        initial_exp_avg=initial_m,
        initial_exp_avg_sq=initial_v,
    )
    base_state = probe.encode_initial_state().detach()
    with torch.enable_grad():
        next_state = probe.map(base_state).detach()
    scale = probe.coordinate_scale(base_state)
    normalized_base = scale * base_state
    normalized_next = scale * next_state
    basis = build_semantic_basis(
        probe.layout,
        normalized_base,
        normalized_next,
        components=config.basis.components,
        random_vectors_per_block=config.basis.random_vectors_per_block,
        rank=config.basis.rank,
        seed=config.basis.seed,
        tolerance=config.basis.orthogonalization_tolerance,
    )

    if config.jacobian.mode == "autodiff":
        delta_map = probe.scaled_delta_map(base_state, scale)
        zero = torch.zeros_like(base_state)

        def jvp(direction):
            _, tangent = torch.func.jvp(delta_map, (zero,), (direction,))
            return tangent.detach()

    elif config.jacobian.mode == "finite_difference":
        delta_map = probe.scaled_delta_map(base_state, scale)

        def jvp(direction):
            norm = direction.norm().clamp_min(1e-30)
            step = max(
                config.jacobian.finite_difference_absolute_floor,
                config.jacobian.finite_difference_relative_step / float(norm),
            )
            with torch.enable_grad():
                plus = delta_map(step * direction)
                minus = delta_map(-step * direction)
            return ((plus - minus) / (2.0 * step)).detach()

    else:
        raise ValueError(f"unsupported jacobian mode: {config.jacobian.mode}")

    reduced, projection = project_jacobian(jvp, basis)
    records = analyze_reduced_operator(
        reduced,
        basis,
        phase_count=config.sweep.phase_count,
        finite_horizon=config.sweep.finite_horizon,
        compute_kreiss=config.sweep.compute_kreiss,
        maximum_couplings=config.sweep.maximum_couplings,
    )

    root = Path(config.output.root) / config.output.run_name / config.model.run / revision
    manifest = {
        "schema_version": 1,
        "subject": "pythia",
        "run_spec": spec.to_dict(),
        "model_id": model_id,
        "revision": revision,
        "selected_parameters": list(selected_names),
        "selected_parameter_numel": selected_numel,
        "state_numel": probe.layout.state_numel,
        "basis_rank": len(basis),
        "projection": projection.to_dict(),
        "native_state": native_metadata,
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": str(dtype),
        },
        "elapsed_seconds": time.time() - started,
    }
    save_analysis(root, reduced, basis, records, manifest)
    return root


def _align_native_names(state: dict[str, Any], selected_names: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in selected_names:
        candidates = (name, f"module.{name}", name.removeprefix("gpt_neox."))
        match = next((candidate for candidate in candidates if candidate in state), None)
        if match is None:
            suffix = [key for key in state if key.endswith(name)]
            if len(suffix) == 1:
                match = suffix[0]
        if match is None:
            raise KeyError(f"could not align native optimizer state for {name}")
        output[name] = state[match]
    return output


def run_longitudinal(config: PythiaProbeConfig, revisions: list[str]) -> Path:
    roots: list[str] = []
    for revision in revisions:
        payload = config.to_dict()
        payload["model"]["revision"] = revision
        from .config import _construct, BasisConfig, DataConfig, JacobianConfig, ModelConfig, OutputConfig, StateConfig, SweepConfig

        current = PythiaProbeConfig(
            model=_construct(ModelConfig, payload["model"]),
            data=_construct(DataConfig, payload["data"]),
            state=_construct(StateConfig, payload["state"]),
            basis=_construct(BasisConfig, payload["basis"]),
            jacobian=_construct(JacobianConfig, payload["jacobian"]),
            sweep=_construct(SweepConfig, payload["sweep"]),
            output=_construct(OutputConfig, payload["output"]),
        )
        roots.append(str(run_probe(current)))
    summary = Path(config.output.root) / config.output.run_name / config.model.run / "longitudinal.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"roots": roots}, indent=2), encoding="utf-8")
    return summary
