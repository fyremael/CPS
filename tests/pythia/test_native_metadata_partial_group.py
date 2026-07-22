from collections import OrderedDict

import pytest
import torch

from cps.pythia.native_state_packet import reconstruct_zero_adam_state


def _save_packet(tmp_path):
    shapes = OrderedDict(
        [
            ("gpt_neox.layers.0.attention.dense.weight", (2, 3)),
            ("gpt_neox.layers.0.attention.dense.bias", (2,)),
            ("gpt_neox.layers.0.input_layernorm.weight", (2,)),
        ]
    )
    torch.save(
        {"iteration": 143000, "param_shapes": [shapes]},
        tmp_path / "mp_rank_00_model_states.pt",
    )

    full_weight = torch.tensor(
        [0.5, -1.0, 2.0, 3.0, -4.0, 1.5, 0.25, -0.5, 1.25, -1.5]
    )
    decay_m = torch.arange(1.0, 7.0)
    decay_v = decay_m.square()
    for rank in range(2):
        full_slice = slice(rank * 5, (rank + 1) * 5)
        decay_slice = slice(rank * 3, (rank + 1) * 3)
        payload = {
            "optimizer_state_dict": {
                "single_partition_of_fp32_groups": [
                    full_weight[full_slice].clone()
                ],
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": [decay_m[decay_slice].clone()],
                            "exp_avg_sq": [decay_v[decay_slice].clone()],
                        }
                    }
                },
            }
        }
        torch.save(
            payload,
            tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )
    return decay_m


def test_reconciles_checkpoint_param_shapes_with_decay_only_moments(tmp_path):
    decay_m = _save_packet(tmp_path)
    target = "gpt_neox.layers.0.attention.dense.weight"

    state = reconstruct_zero_adam_state(tmp_path, parameter_names=[target])

    assert (
        state.shape_source
        == "model_checkpoint_param_shapes_reconciled_with_native_moment_capacity"
    )
    report = state.shape_validation["optimizer_groups"]
    assert report["coverage"] == "weight_decay_only"
    assert report["expected_group_numel"] == [6]
    assert report["observed_group_numel"] == [6]
    assert report["unavailable_parameter_numel"] == 4
    assert torch.equal(state.exp_avg[target], decay_m.reshape(2, 3))


def test_metadata_partial_packet_fails_closed_for_unavailable_parameter(tmp_path):
    _save_packet(tmp_path)

    with pytest.raises(KeyError, match="requested parameters absent from native state"):
        reconstruct_zero_adam_state(
            tmp_path,
            parameter_names=["gpt_neox.layers.0.attention.dense.bias"],
        )
