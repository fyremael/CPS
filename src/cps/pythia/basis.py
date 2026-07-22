from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .state_layout import StateLayout


@dataclass(frozen=True)
class BasisVector:
    name: str
    vector: object
    parameter_name: str
    component: str


def _normalize(vector, tolerance: float):
    norm = vector.norm()
    if float(norm) <= tolerance:
        return None
    return vector / norm


def orthonormalize(candidates: Iterable[BasisVector], tolerance: float = 1e-7):
    output: list[BasisVector] = []
    for candidate in candidates:
        vector = candidate.vector.clone()
        for existing in output:
            vector = vector - (existing.vector.conj() @ vector) * existing.vector
        normalized = _normalize(vector, tolerance)
        if normalized is None:
            continue
        output.append(
            BasisVector(
                name=candidate.name,
                vector=normalized,
                parameter_name=candidate.parameter_name,
                component=candidate.component,
            )
        )
    return tuple(output)


def build_semantic_basis(
    layout: StateLayout,
    base_state,
    next_state,
    *,
    components: tuple[str, ...],
    random_vectors_per_block: int,
    rank: int,
    seed: int,
    tolerance: float,
):
    import torch

    generator = torch.Generator(device=base_state.device)
    generator.manual_seed(seed)
    candidates: list[BasisVector] = []
    delta = next_state - base_state
    for item in layout.tensors:
        for component in components:
            if component == "theta":
                vector = torch.zeros_like(base_state)
                vector[item.theta] = base_state[item.theta]
                candidates.append(BasisVector(f"{item.name}:theta", vector, item.name, component))
            elif component == "momentum":
                vector = torch.zeros_like(base_state)
                vector[item.momentum] = base_state[item.momentum]
                if float(vector.norm()) == 0.0:
                    vector[item.momentum] = delta[item.momentum]
                candidates.append(
                    BasisVector(f"{item.name}:momentum", vector, item.name, component)
                )
            elif component == "second_moment":
                vector = torch.zeros_like(base_state)
                centered = base_state[item.log_second_moment]
                centered = centered - centered.mean()
                vector[item.log_second_moment] = centered
                candidates.append(
                    BasisVector(f"{item.name}:second_moment", vector, item.name, component)
                )
            elif component == "update":
                vector = torch.zeros_like(base_state)
                vector[item.theta] = delta[item.theta]
                vector[item.momentum] = delta[item.momentum]
                vector[item.log_second_moment] = delta[item.log_second_moment]
                candidates.append(BasisVector(f"{item.name}:update", vector, item.name, component))
            elif component == "random":
                for index in range(random_vectors_per_block):
                    vector = torch.zeros_like(base_state)
                    for part in (item.theta, item.momentum, item.log_second_moment):
                        sample = torch.randint(
                            0,
                            2,
                            (part.stop - part.start,),
                            generator=generator,
                            device=base_state.device,
                        )
                        vector[part] = sample.to(base_state.dtype).mul_(2).sub_(1)
                    candidates.append(
                        BasisVector(
                            f"{item.name}:random{index}",
                            vector,
                            item.name,
                            component,
                        )
                    )
            else:
                raise ValueError(f"unknown basis component: {component}")
    basis = orthonormalize(candidates, tolerance=tolerance)
    if not basis:
        raise RuntimeError("basis construction produced no nonzero directions")
    return basis[:rank]
