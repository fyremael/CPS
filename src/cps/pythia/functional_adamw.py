from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .state_layout import StateLayout


@dataclass(frozen=True)
class AdamWHyperparameters:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    step: int
    v_floor: float = 1e-12

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("Adam betas must lie in [0, 1)")
        if self.epsilon <= 0 or self.v_floor <= 0:
            raise ValueError("epsilon and v_floor must be positive")
        if self.step < 1:
            raise ValueError("step must be at least one")


class FunctionalAdamWProbe:
    """Differentiable selected-coordinate AdamW training map.

    Parameters outside ``selected_names`` are held fixed. Gradients of the loss
    still depend on the full forward pass. The map acts on a flat state vector
    ``(theta, m, log(v + floor))`` for the selected parameters and returns the
    corresponding state after one deterministic optimizer step.
    """

    def __init__(
        self,
        model: Any,
        batch: Mapping[str, Any],
        selected_names: tuple[str, ...],
        hyperparameters: AdamWHyperparameters,
        *,
        initial_exp_avg: Mapping[str, Any] | None = None,
        initial_exp_avg_sq: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for FunctionalAdamWProbe") from exc

        hyperparameters.validate()
        self.torch = torch
        self.model = model
        self.model.eval()
        self.batch = dict(batch)
        self.hyper = hyperparameters
        all_params = dict(model.named_parameters())
        missing = set(selected_names) - set(all_params)
        if missing:
            raise KeyError(f"selected parameters not found: {sorted(missing)}")
        self.selected_names = selected_names
        self.selected = {name: all_params[name].detach() for name in selected_names}
        self.fixed = {name: p.detach() for name, p in all_params.items() if name not in self.selected}
        self.buffers = {name: b.detach() for name, b in model.named_buffers()}
        self.layout = StateLayout.from_parameters(self.selected)
        self.initial_exp_avg = {
            name: (
                initial_exp_avg[name].detach()
                if initial_exp_avg is not None and name in initial_exp_avg
                else torch.zeros_like(tensor)
            )
            for name, tensor in self.selected.items()
        }
        self.initial_exp_avg_sq = {
            name: (
                initial_exp_avg_sq[name].detach().clamp_min(hyperparameters.v_floor)
                if initial_exp_avg_sq is not None and name in initial_exp_avg_sq
                else torch.full_like(tensor, hyperparameters.v_floor)
            )
            for name, tensor in self.selected.items()
        }

    @property
    def device(self) -> Any:
        return next(iter(self.selected.values())).device

    @property
    def dtype(self) -> Any:
        return next(iter(self.selected.values())).dtype

    def encode_initial_state(self) -> Any:
        torch = self.torch
        theta = torch.cat([self.selected[item.name].reshape(-1) for item in self.layout.tensors])
        momentum = torch.cat(
            [self.initial_exp_avg[item.name].reshape(-1) for item in self.layout.tensors]
        )
        log_v = torch.cat(
            [
                torch.log(self.initial_exp_avg_sq[item.name] + self.hyper.v_floor).reshape(-1)
                for item in self.layout.tensors
            ]
        )
        return torch.cat((theta, momentum, log_v))

    def decode_state(self, state: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        torch = self.torch
        if state.ndim != 1 or state.numel() != self.layout.state_numel:
            raise ValueError(
                f"state has shape {tuple(state.shape)}; expected ({self.layout.state_numel},)"
            )
        theta: dict[str, Any] = {}
        momentum: dict[str, Any] = {}
        second: dict[str, Any] = {}
        for item in self.layout.tensors:
            theta[item.name] = state[item.theta].reshape(item.shape)
            momentum[item.name] = state[item.momentum].reshape(item.shape)
            log_v = state[item.log_second_moment].reshape(item.shape)
            second[item.name] = torch.exp(log_v).sub(self.hyper.v_floor).clamp_min(0.0)
        return theta, momentum, second

    def _loss(self, selected_tuple: tuple[Any, ...]) -> Any:
        torch = self.torch
        selected = dict(zip(self.selected_names, selected_tuple, strict=True))
        params = {**self.fixed, **selected}
        outputs = torch.func.functional_call(self.model, (params, self.buffers), (), self.batch)
        if getattr(outputs, "loss", None) is not None:
            return outputs.loss
        logits = outputs.logits
        labels = self.batch.get("labels", self.batch["input_ids"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def map(self, state: Any) -> Any:
        torch = self.torch
        theta, momentum, second = self.decode_state(state)
        theta_tuple = tuple(theta[name] for name in self.selected_names)
        grads = torch.func.grad(self._loss)(theta_tuple)
        beta1, beta2 = self.hyper.beta1, self.hyper.beta2
        bias1 = 1.0 - beta1**self.hyper.step
        bias2 = 1.0 - beta2**self.hyper.step

        next_theta: list[Any] = []
        next_momentum: list[Any] = []
        next_log_v: list[Any] = []
        for name, parameter, gradient in zip(self.selected_names, theta_tuple, grads, strict=True):
            m_new = beta1 * momentum[name] + (1.0 - beta1) * gradient
            v_new = beta2 * second[name] + (1.0 - beta2) * gradient.square()
            m_hat = m_new / bias1
            v_hat = v_new / bias2
            update = m_hat / (torch.sqrt(v_hat) + self.hyper.epsilon)
            parameter_new = parameter * (1.0 - self.hyper.learning_rate * self.hyper.weight_decay)
            parameter_new = parameter_new - self.hyper.learning_rate * update
            next_theta.append(parameter_new.reshape(-1))
            next_momentum.append(m_new.reshape(-1))
            next_log_v.append(torch.log(v_new + self.hyper.v_floor).reshape(-1))
        return torch.cat((*next_theta, *next_momentum, *next_log_v))

    def coordinate_scale(self, state: Any, epsilon: float = 1e-8) -> Any:
        """Diagonal similarity scaling for dimensionless optimizer coordinates."""

        torch = self.torch
        theta, momentum, _ = self.decode_state(state)
        scale = torch.ones_like(state)
        for item in self.layout.tensors:
            theta_rms = theta[item.name].square().mean().sqrt().clamp_min(epsilon)
            momentum_rms = momentum[item.name].square().mean().sqrt().clamp_min(epsilon)
            scale[item.theta] = theta_rms.reciprocal()
            scale[item.momentum] = momentum_rms.reciprocal()
            scale[item.log_second_moment] = 1.0
        return scale

    def scaled_delta_map(self, base_state: Any, scale: Any | None = None):
        if scale is None:
            scale = self.coordinate_scale(base_state)
        next_base = self.map(base_state).detach()

        def delta_map(delta: Any) -> Any:
            physical = base_state + delta / scale
            return scale * (self.map(physical) - next_base)

        return delta_map

    def jvp(self, base_state: Any, direction: Any, *, scaled: bool = True) -> Any:
        torch = self.torch
        if scaled:
            function = self.scaled_delta_map(base_state)
            zero = torch.zeros_like(base_state)
            _, tangent = torch.func.jvp(function, (zero,), (direction,))
            return tangent
        _, tangent = torch.func.jvp(self.map, (base_state,), (direction,))
        return tangent
