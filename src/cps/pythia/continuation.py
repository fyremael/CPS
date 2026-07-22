from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cps.progress import ConsoleReporter


@dataclass(frozen=True)
class ContinuationControl:
    name: str = "baseline"
    learning_rate_scale: float = 1.0
    beta1: float | None = None
    beta2: float | None = None
    epsilon_scale: float = 1.0
    gradient_clip: float | None = None


@dataclass(frozen=True)
class ContinuationConfig:
    model_id: str = "EleutherAI/pythia-70m"
    revision: str = "step1000"
    device: str = "auto"
    dtype: str = "float32"
    steps: int = 20
    sequence_length: int = 64
    batch_size: int = 1
    seed: int = 101
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    baseline: ContinuationControl = field(default_factory=ContinuationControl)
    intervention: ContinuationControl = field(
        default_factory=lambda: ContinuationControl(name="cps", learning_rate_scale=0.8)
    )
    output_dir: str = "artifacts/pythia/continuation"
    verbose: bool = True


@dataclass(frozen=True)
class StepRecord:
    step: int
    loss: float
    gradient_norm: float
    parameter_norm: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ForkResult:
    control: dict[str, Any]
    records: tuple[StepRecord, ...]
    final_loss: float
    maximum_gradient_norm: float
    loss_spike: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "records": [asdict(record) for record in self.records],
            "final_loss": self.final_loss,
            "maximum_gradient_norm": self.maximum_gradient_norm,
            "loss_spike": self.loss_spike,
        }


def run_matched_continuation(
    config: ContinuationConfig,
    *,
    reporter: ConsoleReporter | None = None,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra for continuation experiments") from exc

    reporter = reporter or ConsoleReporter(enabled=config.verbose, prefix="CPS-CONT")
    reporter.title("Coupling-Phase Spectroscopy · matched continuation")
    if config.steps < 1:
        raise ValueError("steps must be positive")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        config.dtype
    ]
    if device.type == "cpu":
        dtype = torch.float32

    reporter.section("Load the common checkpoint and deterministic batches")
    reporter.metric("model", config.model_id)
    reporter.metric("revision", config.revision)
    reporter.metric("steps per fork", config.steps)
    reporter.metric("device", device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=dtype,
    ).to(device)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    batches = _make_batches(tokenizer, config, device)
    reporter.metric("prepared batches", len(batches))

    results: dict[str, ForkResult] = {}
    for fork_index, control in enumerate((config.baseline, config.intervention), start=1):
        reporter.section(f"Run matched fork {fork_index}/2: {control.name}")
        reporter.info(
            f"lr scale={control.learning_rate_scale}, beta1={control.beta1}, "
            f"beta2={control.beta2}, epsilon scale={control.epsilon_scale}"
        )
        model.load_state_dict(initial)
        model.train()
        results[control.name] = _run_fork(model, batches, config, control, reporter)

    output = Path(config.output_dir) / config.revision
    output.mkdir(parents=True, exist_ok=True)
    target = output / "matched_continuation.json"
    target.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "results": {name: result.to_dict() for name, result in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    baseline = results[config.baseline.name]
    intervention = results[config.intervention.name]
    reporter.section("Compare the matched outcomes")
    reporter.metric("baseline final loss", f"{baseline.final_loss:.6g}")
    reporter.metric("intervention final loss", f"{intervention.final_loss:.6g}")
    reporter.metric(
        "final-loss difference (intervention - baseline)",
        f"{intervention.final_loss - baseline.final_loss:+.6g}",
    )
    reporter.metric("evidence packet", target)
    reporter.success("Matched continuation completed.")
    return target


def _make_batches(tokenizer, config: ContinuationConfig, device):
    import torch

    corpus = [
        "A stable numerical method must control both local error and accumulated error.",
        "Optimization is the design of a dynamical system whose fixed points solve a task.",
        "The spectrum describes modes, while nonnormality describes how modes cooperate.",
        "A diagnostic becomes useful when it selects an intervention that survives a control test.",
    ]
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batches = []
    for step in range(config.steps):
        index = int(torch.randint(0, len(corpus), (1,), generator=generator))
        encoded = tokenizer(
            [corpus[index]] * config.batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.sequence_length,
        )
        input_ids = encoded["input_ids"].to(device)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"].to(device),
                "labels": input_ids.clone(),
            }
        )
    return batches


def _run_fork(
    model,
    batches,
    config: ContinuationConfig,
    control: ContinuationControl,
    reporter: ConsoleReporter,
) -> ForkResult:
    import torch

    beta1 = config.beta1 if control.beta1 is None else control.beta1
    beta2 = config.beta2 if control.beta2 is None else control.beta2
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * control.learning_rate_scale,
        betas=(beta1, beta2),
        eps=config.epsilon * control.epsilon_scale,
        weight_decay=config.weight_decay,
    )
    records: list[StepRecord] = []
    started = time.time()
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        grad_sq = torch.zeros((), device=loss.device)
        parameter_sq = torch.zeros((), device=loss.device)
        for parameter in model.parameters():
            parameter_sq += parameter.detach().float().square().sum()
            if parameter.grad is not None:
                grad_sq += parameter.grad.detach().float().square().sum()
        gradient_norm = float(grad_sq.sqrt())
        if control.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), control.gradient_clip)
        optimizer.step()
        record = StepRecord(
            step=step,
            loss=float(loss.detach()),
            gradient_norm=gradient_norm,
            parameter_norm=float(parameter_sq.sqrt()),
            elapsed_seconds=time.time() - started,
        )
        records.append(record)
        reporter.progress(
            step,
            len(batches),
            control.name,
            f"loss={record.loss:.5f}; ||g||={record.gradient_norm:.4g}; "
            f"elapsed={record.elapsed_seconds:.1f}s",
        )
    losses = [record.loss for record in records]
    trend = min(losses[0], sum(losses[: min(3, len(losses))]) / min(3, len(losses)))
    return ForkResult(
        control=asdict(control),
        records=tuple(records),
        final_loss=losses[-1],
        maximum_gradient_norm=max(record.gradient_norm for record in records),
        loss_spike=max(losses) - trend,
    )
