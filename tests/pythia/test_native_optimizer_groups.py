from collections import OrderedDict

import pytest
import torch

from cps.pythia.native_state import reconstruct_zero_adam_state


def _save_two_group_checkpoint(tmp_path, *, decay_w, no_decay_w, decay_m, no_decay_m):
    torch.save({"iteration": 143000}, tmp_path / "mp_rank_00_model_states.pt")
    for rank in range(2):
        decay_slice = slice(rank * 3, (rank + 1) * 3)
        no_decay_slice = slice(rank * 2, (rank + 1) * 2)
        payload = {
            "optimizer_state_dict": {
                "single_partition_of_fp32_groups": [
                    decay_w[decay_slice].clone(),
                    no_decay_w[no_decay_slice].clone(),
                ],
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": decay_m[decay_slice].clone(),
                            "exp_avg_sq": decay_m[decay_slice].square(),
                        },
                        1: {
                            "exp_avg": no_decay_m[no_decay_slice].clone(),
                            "exp_avg_sq": no_decay_m[no_decay_slice].square(),
                        },
                    }
                },
            }
        }
        torch.save(
            payload,
            tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )


def test_infers_weight_decay_and_no_decay_groups(tmp_path):
    shapes = OrderedDict(
        [
            ("gpt_neox.layers.0.attention.dense.weight", (2, 3)),
            ("gpt_neox.layers.0.attention.dense.bias", (2,)),
            ("gpt_neox.layers.0.input_layernorm.weight", (2,)),
        ]
    )
    decay_w = torch.tensor([0.5, -1.0, 2.0, 3.0, -4.0, 1.5])
    no_decay_w = torch.tensor([0.25, -0.5, 1.25, -1.5])
    decay_m = torch.arange(1.0, 7.0)
    no_decay_m = torch.arange(7.0, 11.0)
    _save_two_group_checkpoint(
        tmp_path,
        decay_w=decay_w,
        no_decay_w=no_decay_w,
        decay_m=decay_m,
        no_decay_m=no_decay_m,
    )

    references = {
        "gpt_neox.layers.0.attention.dense.weight": {
            "l2": float(decay_w.norm()),
            "l1": float(decay_w.abs().sum()),
        },
        "gpt_neox.layers.0.attention.dense.bias": {
            "l2": float(no_decay_w[:2].norm()),
            "l1": float(no_decay_w[:2].abs().sum()),
        },
        "gpt_neox.layers.0.input_layernorm.weight": {
            "l2": float(no_decay_w[2:].norm()),
            "l1": float(no_decay_w[2:].abs().sum()),
        },
    }

    state = reconstruct_zero_adam_state(
        tmp_path,
        parameter_shapes=shapes,
        reference_signatures=references,
    )

    assert (
        state.shape_source
        == "validated_model_parameter_order_with_inferred_optimizer_groups"
    )
    groups = state.shape_validation["optimizer_groups"]
    assert groups["order"] == "weight_decay_then_no_decay"
    assert groups["expected_group_numel"] == [6, 4]
    assert groups["observed_group_numel"] == [6, 4]
    assert groups["padding_numel"] == [0, 0]
    assert torch.equal(
        state.exp_avg["gpt_neox.layers.0.attention.dense.weight"],
        decay_m.reshape(2, 3),
    )
    assert torch.equal(
        state.exp_avg["gpt_neox.layers.0.attention.dense.bias"],
        no_decay_m[:2],
    )
    assert torch.equal(
        state.exp_avg["gpt_neox.layers.0.input_layernorm.weight"],
        no_decay_m[2:],
    )


def test_rejects_optimizer_group_capacity_mismatch(tmp_path):
    shapes = OrderedDict(
        [
            ("linear.weight", (2, 3)),
            ("linear.bias", (2,)),
            ("norm.weight", (2,)),
        ]
    )
    decay_w = torch.arange(1.0, 7.0)
    no_decay_w = torch.arange(7.0, 11.0)
    decay_m = torch.arange(1.0, 7.0)
    no_decay_m = torch.arange(7.0, 11.0)
    _save_two_group_checkpoint(
        tmp_path,
        decay_w=decay_w,
        no_decay_w=no_decay_w,
        decay_m=decay_m,
        no_decay_m=no_decay_m,
    )

    wrong_shapes = OrderedDict(shapes)
    wrong_shapes["extra.weight"] = (3, 3)
    references = {
        name: {"l2": 1.0, "l1": 1.0}
        for name in wrong_shapes
    }

    with pytest.raises(ValueError, match="does not fit the native optimizer moment groups"):
        reconstruct_zero_adam_state(
            tmp_path,
            parameter_shapes=wrong_shapes,
            reference_signatures=references,
        )


def test_pythia_70m_deficit_is_exactly_no_decay_group():
    hidden = 512
    intermediate = 2048
    shapes = OrderedDict()
    shapes["gpt_neox.embed_in.weight"] = (50304, hidden)
    for layer in range(6):
        prefix = f"gpt_neox.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.input_layernorm.bias"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.bias"] = (hidden,)
        shapes[f"{prefix}.attention.query_key_value.weight"] = (
            3 * hidden,
            hidden,
        )
        shapes[f"{prefix}.attention.query_key_value.bias"] = (3 * hidden,)
        shapes[f"{prefix}.attention.dense.weight"] = (hidden, hidden)
        shapes[f"{prefix}.attention.dense.bias"] = (hidden,)
        shapes[f"{prefix}.mlp.dense_h_to_4h.weight"] = (
            intermediate,
            hidden,
        )
        shapes[f"{prefix}.mlp.dense_h_to_4h.bias"] = (intermediate,)
        shapes[f"{prefix}.mlp.dense_4h_to_h.weight"] = (
            hidden,
            intermediate,
        )
        shapes[f"{prefix}.mlp.dense_4h_to_h.bias"] = (hidden,)
    shapes["gpt_neox.final_layer_norm.weight"] = (hidden,)
    shapes["gpt_neox.final_layer_norm.bias"] = (hidden,)
    shapes["embed_out.weight"] = (50304, hidden)

    no_decay = sum(
        torch.tensor(shape).prod().item()
        for shape in shapes.values()
        if len(shape) <= 1
    )
    decay = sum(
        torch.tensor(shape).prod().item()
        for shape in shapes.values()
        if len(shape) > 1
    )

    assert no_decay == 40960
    assert decay == 70385664
    assert decay + no_decay == 70426624
