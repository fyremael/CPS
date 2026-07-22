from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cps.progress import NullReporter
from cps.pythia.config import JacobianConfig
from cps.pythia.jvp import build_jvp


def _quadratic_delta_map(base):
    def function(delta):
        return (base + delta).square() - base.square()

    return function


def test_forward_ad_jvp_preflight_reports_exact_backend() -> None:
    base = torch.tensor([2.0, -1.0])
    zero = torch.zeros_like(base)
    direction = torch.tensor([0.5, 2.0])
    jvp, diagnostics = build_jvp(
        _quadratic_delta_map(base),
        zero,
        direction,
        JacobianConfig(mode="autodiff", autodiff_backend="auto"),
        reporter=NullReporter(),
    )
    expected = 2.0 * base * direction
    assert torch.allclose(jvp(direction), expected)
    assert diagnostics.effective_backend == "torch.func.jvp"
    assert diagnostics.fallback_used is False


def test_auto_backend_falls_back_when_forward_ad_is_unimplemented(monkeypatch) -> None:
    base = torch.tensor([2.0, -1.0])
    zero = torch.zeros_like(base)
    direction = torch.tensor([0.5, 2.0])

    def unavailable(*args, **kwargs):
        raise NotImplementedError("Trying to use forward AD with a kernel that does not support it")

    monkeypatch.setattr(torch.func, "jvp", unavailable)
    jvp, diagnostics = build_jvp(
        _quadratic_delta_map(base),
        zero,
        direction,
        JacobianConfig(
            mode="autodiff",
            autodiff_backend="auto",
            fallback_to_finite_difference=True,
            finite_difference_relative_step=1e-4,
            finite_difference_absolute_floor=1e-6,
        ),
        reporter=NullReporter(),
    )
    expected = 2.0 * base * direction
    assert torch.allclose(jvp(direction), expected, rtol=2e-3, atol=2e-3)
    assert diagnostics.effective_backend == "centered_difference"
    assert diagnostics.fallback_used is True
    assert "NotImplementedError" in (diagnostics.fallback_reason or "")


def test_explicit_forward_ad_backend_fails_closed(monkeypatch) -> None:
    base = torch.tensor([2.0, -1.0])
    zero = torch.zeros_like(base)
    direction = torch.tensor([0.5, 2.0])

    def unavailable(*args, **kwargs):
        raise NotImplementedError("forward AD is not implemented")

    monkeypatch.setattr(torch.func, "jvp", unavailable)
    with pytest.raises(RuntimeError, match="attention_implementation=eager"):
        build_jvp(
            _quadratic_delta_map(base),
            zero,
            direction,
            JacobianConfig(
                mode="autodiff",
                autodiff_backend="forward_ad",
                fallback_to_finite_difference=True,
            ),
            reporter=NullReporter(),
        )
