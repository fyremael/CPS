from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from cps.progress import ConsoleReporter, NullReporter

from .config import JacobianConfig


@dataclass(frozen=True)
class JVPDiagnostics:
    requested_mode: str
    requested_backend: str
    effective_mode: str
    effective_backend: str
    fallback_used: bool
    fallback_reason: str | None
    preflight_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_difference_jvp(
    delta_map: Callable[[Any], Any],
    config: JacobianConfig,
) -> Callable[[Any], Any]:
    def jvp(direction: Any) -> Any:
        norm = direction.norm().clamp_min(1e-30)
        step = max(
            config.finite_difference_absolute_floor,
            config.finite_difference_relative_step / float(norm),
        )
        import torch

        with torch.enable_grad():
            plus = delta_map(step * direction)
            minus = delta_map(-step * direction)
        return ((plus - minus) / (2.0 * step)).detach()

    return jvp


def _forward_ad_jvp(delta_map: Callable[[Any], Any], zero: Any) -> Callable[[Any], Any]:
    import torch

    def jvp(direction: Any) -> Any:
        _, tangent = torch.func.jvp(delta_map, (zero,), (direction,))
        return tangent.detach()

    return jvp


def _is_forward_ad_limitation(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    signatures = (
        "forward ad",
        "forward-mode ad",
        "forward mode ad",
        "does not support it because it has not been implemented",
        "derivatives.yaml",
        "jvp is not implemented",
    )
    return isinstance(error, NotImplementedError) or any(item in text for item in signatures)


def build_jvp(
    delta_map: Callable[[Any], Any],
    zero: Any,
    probe_direction: Any,
    config: JacobianConfig,
    *,
    reporter: ConsoleReporter | None = None,
) -> tuple[Callable[[Any], Any], JVPDiagnostics]:
    """Build and preflight the Jacobian-vector product implementation.

    ``torch.func.jvp`` is preferred because it computes exact forward-mode JVPs.
    Some fused kernels, notably efficient SDPA kernels, do not yet implement
    forward AD. In ``auto`` mode CPS records that limitation and falls back to a
    centered finite difference rather than failing midway through a notebook.
    The manifest always records which backend actually produced the operator.
    """

    reporter = reporter or NullReporter()
    if config.mode == "finite_difference":
        candidate = _finite_difference_jvp(delta_map, config)
        image = candidate(probe_direction)
        return candidate, JVPDiagnostics(
            requested_mode=config.mode,
            requested_backend=config.autodiff_backend,
            effective_mode="finite_difference",
            effective_backend="centered_difference",
            fallback_used=False,
            fallback_reason=None,
            preflight_norm=float(image.norm()),
        )

    if config.mode != "autodiff":
        raise ValueError(f"unsupported jacobian mode: {config.mode}")
    if config.autodiff_backend not in {"auto", "forward_ad"}:
        raise ValueError(f"unsupported autodiff_backend: {config.autodiff_backend}")

    candidate = _forward_ad_jvp(delta_map, zero)
    try:
        image = candidate(probe_direction)
        return candidate, JVPDiagnostics(
            requested_mode=config.mode,
            requested_backend=config.autodiff_backend,
            effective_mode="autodiff",
            effective_backend="torch.func.jvp",
            fallback_used=False,
            fallback_reason=None,
            preflight_norm=float(image.norm()),
        )
    except (NotImplementedError, RuntimeError) as error:
        if not _is_forward_ad_limitation(error):
            raise
        if config.autodiff_backend == "forward_ad" or not config.fallback_to_finite_difference:
            raise RuntimeError(
                "The selected model kernel does not implement forward AD. Set "
                "model.attention_implementation=eager or enable "
                "jacobian.fallback_to_finite_difference."
            ) from error
        reason = f"{type(error).__name__}: {error}"
        reporter.warning(
            "Exact forward-mode JVP is unavailable for an active kernel. "
            "Switching to the preregistered centered finite-difference fallback."
        )
        fallback = _finite_difference_jvp(delta_map, config)
        image = fallback(probe_direction)
        return fallback, JVPDiagnostics(
            requested_mode=config.mode,
            requested_backend=config.autodiff_backend,
            effective_mode="finite_difference",
            effective_backend="centered_difference",
            fallback_used=True,
            fallback_reason=reason,
            preflight_norm=float(image.norm()),
        )
