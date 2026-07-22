from collections import OrderedDict

import pytest
import torch

from cps.pythia.native_state import reconstruct_zero_adam_state


def _save_single_moment_group_checkpoint(tmp_path):
    torch.save({"iteration": 143000}, tmp_path / "mp_rank_00_model_states.pt")

    # Model order: one matrix parameter followed by two one-dimensional tensors.
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
                            # Legacy packets may wrap a flattened moment in a list.
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

    shapes = OrderedDict(
        [
            ("gpt_neox.layers.0.attention.dense.weight", (2, 3)),
            ("gpt_neox.layers.0.attention.dense.bias", (2,)),
            ("gpt_neox.layers.0.input_layernorm.weight", (2,)),
        ]
    )
    references = {
        "gpt_neox.layers.0.attention.dense.weight": {
            "l2": float(full_weight[:6].norm()),
            "l1": float(full_weight[:6].abs().sum()),
        },
        "gpt_neox.layers.0.attention.dense.bias": {
            "l2": float(full_weight[6:8].norm()),
            "l1": float(full_weight[6:8].abs().sum()),
        },
        "gpt_neox.layers.0.input_layernorm.weight": {
            "l2": float(full_weight[8:].norm()),
            "l1": float(full_weight[8:].abs().sum()),
        },
    }
    return shapes, references, decay_m


def test_reconstructs_requested_decay_parameter_from_single_moment_group(tmp_path):
    shapes, references, decay_m = _save_single_moment_group_checkpoint(tmp_path)
    target = "gpt_neox.layers.0.attention.dense.weight"

    state = reconstruct_zero_adam_state(
        tmp_path,
        parameter_names=[target],
        parameter_shapes=shapes,
        reference_signatures=references,
    )

    assert (
        state.shape_source
        == "validated_model_parameter_order_with_inferred_optimizer_groups"
    )
    report = state.shape_validation["optimizer_groups"]
    assert report["coverage"] == "weight_decay_only"
    assert report["partial_native_state"] is True
    assert report["expected_group_numel"] == [6]
    assert report["observed_group_numel"] == [6]
    assert report["unavailable_parameter_numel"] == 4
    assert torch.equal(state.exp_avg[target], decay_m.reshape(2, 3))


def test_fails_closed_when_requested_parameter_has_no_native_moments(tmp_path):
    shapes, references, _ = _save_single_moment_group_checkpoint(tmp_path)

    with pytest.raises(KeyError, match="requested parameters absent from native state"):
        reconstruct_zero_adam_state(
            tmp_path,
            parameter_names=["gpt_neox.layers.0.attention.dense.bias"],
            parameter_shapes=shapes,
            reference_signatures=references,
        )
