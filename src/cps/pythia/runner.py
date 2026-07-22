from __future__ import annotations

import json
from collections import OrderedDict
import platform
import time
from pathlib import Path
from typing import Any

from cps.progress import ConsoleReporter

from .analysis import analyze_reduced_operator, save_analysis
from .basis import build_semantic_basis
from .blocks import select_parameter_names
from .config import PythiaProbeConfig
from .data import build_batch
from .functional_adamw import AdamWHyperparameters, FunctionalAdamWProbe
from .jvp import build_jvp
from .native_state import parameter_reference_signatures
from .native_state_packet import reconstruct_zero_adam_state
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


def _load_model(config: PythiaProbeConfig, model_id: str, device, dtype, reporter):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "cache_dir": config.model.cache_dir,
        "torch_dtype": dtype,
        "trust_remote_code": config.model.trust_remote_code,
    }
    requested_attention = config.model.attention_implementation
    if requested_attention != "auto":
        kwargs["attn_implementation"] = requested_attention
    reporter.info(
        f"Loading model with attention implementation={requested_attention!r}; "
        "'eager' avoids fused-SDPA forward-AD gaps."
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError as error:
        if "attn_implementation" not in str(error) or "attn_implementation" not in kwargs:
            raise
        reporter.warning(
            "Installed Transformers does not accept attn_implementation. Retrying with its "
            "default attention backend; the JVP preflight will fall back safely if required."
        )
        kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    model.eval()
    actual_attention = getattr(
        model.config,
        "_attn_implementation",
        getattr(model.config, "attn_implementation", "unknown"),
    )
    return model, str(actual_attention)


def run_probe(
    config: PythiaProbeConfig,
    *,
    reporter: ConsoleReporter | None = None,
) -> Path:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra before running a Pythia probe") from exc

    reporter = reporter or ConsoleReporter(enabled=config.output.verbose, prefix="CPS-PYTHIA")
    reporter.title("Coupling-Phase Spectroscopy · Pythia optimizer-state probe")
    started = time.time()

    reporter.section("Resolve subject, checkpoint, numerical precision, and accelerator")
    spec = get_run_spec(config.model.run)
    model_id = config.model.model_id or spec.model_id
    revision = config.model.revision
    device = _device(config.model.device)
    dtype = _dtype(config.model.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        reporter.warning("CPU execution requested with reduced precision; using float32 instead.")
        dtype = torch.float32
    reporter.metric("subject", config.model.run)
    reporter.metric("model", model_id)
    reporter.metric("revision", revision)
    reporter.metric("device", device)
    reporter.metric("dtype", dtype)

    reporter.section("Load tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, actual_attention = _load_model(config, model_id, device, dtype, reporter)
    reporter.metric("active attention backend", actual_attention)
    reporter.metric("model parameters", f"{sum(p.numel() for p in model.parameters()):,}")

    reporter.section("Select governed parameter coordinates")
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
    reporter.metric("selected tensors", len(selected_names))
    reporter.metric("selected parameters", f"{selected_numel:,}")
    for name in selected_names:
        reporter.info(f"{name}: shape={tuple(all_named[name].shape)}, numel={all_named[name].numel():,}")

    reporter.section("Construct the deterministic probe batch")
    batch = build_batch(tokenizer, config.data, device)
    reporter.metric("batch size", int(batch["input_ids"].shape[0]))
    reporter.metric("sequence length", int(batch["input_ids"].shape[1]))
    reporter.metric("data source", config.data.source)
    reporter.metric("data seed", config.data.seed)

    reporter.section("Reconstruct or load optimizer moments")
    initial_m = None
    initial_v = None
    native_metadata: dict[str, Any] = {"mode": config.state.moment_source}
    if config.state.moment_source == "native":
        if config.state.native_checkpoint_dir is None:
            raise ValueError("native_checkpoint_dir is required for moment_source=native")
        reporter.info(f"Reading ZeRO-partitioned state from {config.state.native_checkpoint_dir}")
        reporter.info(
            "Preparing the full ordered model-shape contract. Historical GPT-NeoX packets may "
            "omit param_shapes; CPS will use this contract only after validating it against the "
            "native fp32 master-weight partitions."
        )
        ordered_shapes = OrderedDict(
            (name, tuple(parameter.shape)) for name, parameter in model.named_parameters()
        )
        signatures = parameter_reference_signatures(all_named)
        native = reconstruct_zero_adam_state(
            config.state.native_checkpoint_dir,
            parameter_names=selected_names,
            parameter_shapes=ordered_shapes,
            reference_signatures=signatures,
        )
        initial_m = _align_native_names(native.exp_avg, selected_names)
        initial_v = _align_native_names(native.exp_avg_sq, selected_names)
        native_metadata.update(
            {
                "partition_count": native.partition_count,
                "source_files": list(native.source_files),
                "parameter_count": native.parameter_count,
                "shape_source": native.shape_source,
                "shape_validation": native.shape_validation,
            }
        )
        reporter.metric("ZeRO partitions", native.partition_count)
        reporter.metric("reconstructed moment parameters", native.parameter_count)
        reporter.metric("native shape source", native.shape_source)
        if native.shape_validation.get("match_fraction") is not None:
            reporter.metric(
                "native order signature match",
                f"{native.shape_validation['match_fraction']:.1%}",
            )
    elif config.state.moment_source not in {"reconstructed", "zero"}:
        raise ValueError(f"unsupported moment_source: {config.state.moment_source}")
    else:
        reporter.info(
            f"Using {config.state.moment_source!r} moments. This is an instrument-validation "
            "state, not an exact historical optimizer-state claim."
        )

    reporter.section("Build the differentiable AdamW state map")
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
    reporter.metric("optimizer-state coordinates", f"{probe.layout.state_numel:,}")
    reporter.info("Evaluating one deterministic optimizer transition; this may take a moment.")
    with torch.enable_grad():
        next_state = probe.map(base_state).detach()
    scale = probe.coordinate_scale(base_state)
    normalized_base = scale * base_state
    normalized_next = scale * next_state
    reporter.metric("normalized state displacement", f"{float((normalized_next-normalized_base).norm()):.6g}")

    reporter.section("Construct the semantic projection basis")
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
    reporter.metric("requested rank", config.basis.rank)
    reporter.metric("realized rank", len(basis))
    for index, item in enumerate(basis):
        reporter.info(f"q[{index:02d}] {item.name}")

    reporter.section("Preflight the Jacobian-vector product backend")
    delta_map = probe.scaled_delta_map(base_state, scale)
    zero = torch.zeros_like(base_state)
    jvp, jvp_diagnostics = build_jvp(
        delta_map,
        zero,
        basis[0].vector,
        config.jacobian,
        reporter=reporter,
    )
    reporter.metric("effective JVP mode", jvp_diagnostics.effective_mode)
    reporter.metric("effective JVP backend", jvp_diagnostics.effective_backend)
    reporter.metric("preflight image norm", f"{jvp_diagnostics.preflight_norm:.6g}")
    if jvp_diagnostics.fallback_used:
        reporter.warning(
            "The run remains valid as a finite-difference probe, but the manifest marks the "
            "fallback so it cannot be mistaken for an exact autodiff operator."
        )

    reporter.section("Project the optimizer-state Jacobian")

    def projection_progress(current, total, source, norm, residual):
        reporter.progress(
            current,
            total,
            "JVP columns",
            f"{source.name}; ||Jq||={norm:.3g}; closure residual={residual:.3g}",
        )

    reduced, projection = project_jacobian(jvp, basis, progress=projection_progress)
    reporter.metric("maximum closure residual", f"{projection.maximum_closure_residual:.6g}")
    reporter.metric("mean closure residual", f"{projection.mean_closure_residual:.6g}")

    reporter.section("Sweep the strongest directed couplings through phase")

    def coupling_progress(current, total, record):
        reporter.progress(
            current,
            total,
            "couplings",
            f"{record.source} → {record.target}; |a|={record.magnitude:.3g}; "
            f"max ρ={record.metrics['spectral_radius_max']:.3g}",
        )

    records = analyze_reduced_operator(
        reduced,
        basis,
        phase_count=config.sweep.phase_count,
        finite_horizon=config.sweep.finite_horizon,
        compute_kreiss=config.sweep.compute_kreiss,
        maximum_couplings=config.sweep.maximum_couplings,
        progress=coupling_progress,
    )

    reporter.section("Write the evidence packet")
    root = Path(config.output.root) / config.output.run_name / config.model.run / revision
    manifest = {
        "schema_version": 2,
        "subject": "pythia",
        "run_spec": spec.to_dict(),
        "model_id": model_id,
        "revision": revision,
        "selected_parameters": list(selected_names),
        "selected_parameter_numel": selected_numel,
        "state_numel": probe.layout.state_numel,
        "basis_rank": len(basis),
        "projection": projection.to_dict(),
        "jacobian": jvp_diagnostics.to_dict(),
        "attention": {
            "requested_implementation": config.model.attention_implementation,
            "active_implementation": actual_attention,
        },
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
    reporter.metric("output directory", root)
    reporter.metric("elapsed", f"{manifest['elapsed_seconds']:.2f}", "seconds")
    if records:
        strongest = max(records, key=lambda item: item.metrics["spectral_radius_max"])
        reporter.info(
            "Largest observed phase-envelope spectral radius: "
            f"{strongest.metrics['spectral_radius_max']:.6g} for "
            f"{strongest.source} → {strongest.target}."
        )
    reporter.success("Pythia CPS probe completed and evidence was written.")
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


def run_longitudinal(
    config: PythiaProbeConfig,
    revisions: list[str],
    *,
    reporter: ConsoleReporter | None = None,
) -> Path:
    reporter = reporter or ConsoleReporter(enabled=config.output.verbose, prefix="CPS-LONG")
    reporter.title("Coupling-Phase Spectroscopy · longitudinal Pythia campaign")
    reporter.metric("checkpoint count", len(revisions))
    roots: list[str] = []
    for index, revision in enumerate(revisions, start=1):
        reporter.section(f"Checkpoint {index}/{len(revisions)}: {revision}")
        payload = config.to_dict()
        payload["model"]["revision"] = revision
        from .config import (
            BasisConfig,
            DataConfig,
            JacobianConfig,
            ModelConfig,
            OutputConfig,
            StateConfig,
            SweepConfig,
            _construct,
        )

        current = PythiaProbeConfig(
            model=_construct(ModelConfig, payload["model"]),
            data=_construct(DataConfig, payload["data"]),
            state=_construct(StateConfig, payload["state"]),
            basis=_construct(BasisConfig, payload["basis"]),
            jacobian=_construct(JacobianConfig, payload["jacobian"]),
            sweep=_construct(SweepConfig, payload["sweep"]),
            output=_construct(OutputConfig, payload["output"]),
        )
        child = ConsoleReporter(enabled=config.output.verbose, prefix=f"CPS-{revision}")
        roots.append(str(run_probe(current, reporter=child)))
        reporter.progress(index, len(revisions), "checkpoints", revision)
    summary = Path(config.output.root) / config.output.run_name / config.model.run / "longitudinal.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"roots": roots}, indent=2), encoding="utf-8")
    reporter.success(f"Longitudinal summary written to {summary}")
    return summary
