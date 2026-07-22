import importlib

import torch

import cps.pythia.native_state_packet as packet
from cps.pythia.native_state_stable import _infer_shape_groups


def test_packet_module_can_be_reloaded_repeatedly():
    current = packet
    for _ in range(4):
        current = importlib.reload(current)
        assert current.reconstruct_zero_adam_state.__module__ == (
            "cps.pythia.native_state_stable"
        )


def test_two_group_fallback_remains_non_recursive_after_reload():
    from collections import OrderedDict

    current = packet
    for _ in range(3):
        current = importlib.reload(current)

    shapes = OrderedDict(
        [
            ("linear.weight", (2, 3)),
            ("linear.bias", (2,)),
            ("norm.weight", (2,)),
        ]
    )
    groups, report = _infer_shape_groups(
        [shapes],
        (6, 4),
        partition_count=1,
    )

    assert report is not None
    assert report["order"] == "weight_decay_then_no_decay"
    assert [sum(torch.tensor(shape).prod().item() for shape in group.values()) for group in groups] == [6, 4]
