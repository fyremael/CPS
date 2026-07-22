import torch

from cps.pythia.state_layout import StateLayout


def test_layout_partitions_three_state_components():
    layout = StateLayout.from_parameters({"a": torch.zeros(2, 3), "b": torch.zeros(4)})
    assert layout.parameter_numel == 10
    assert layout.state_numel == 30
    assert layout.tensor("a").theta == slice(0, 6)
    assert layout.tensor("a").momentum == slice(10, 16)
    assert layout.tensor("b").log_second_moment == slice(26, 30)
