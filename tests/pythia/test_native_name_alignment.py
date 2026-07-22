from collections import OrderedDict

import torch

from cps.pythia.native_state_packet import reconstruct_zero_adam_state
from cps.pythia.native_name_alignment import resolve_requested_parameter_names


def test_maps_hf_layer_name_to_native_pipeline_occurrence():
    native = [
        OrderedDict(
            [
                ("module.sequential.2.attention.dense.weight", (2, 2)),
                ("module.sequential.3.attention.dense.weight", (2, 2)),
            ]
        )
    ]
    caller = OrderedDict(
        [
            ("gpt_neox.layers.0.attention.dense.weight", (2, 2)),
            ("gpt_neox.layers.1.attention.dense.weight", (2, 2)),
        ]
    )

    alignment = resolve_requested_parameter_names(
        ["gpt_neox.layers.0.attention.dense.weight"],
        native,
        caller,
    )

    assert alignment.requested_to_native == {
        "gpt_neox.layers.0.attention.dense.weight": (
            "module.sequential.2.attention.dense.weight"
        )
    }
    assert alignment.report["alias_count"] == 1
    assert alignment.report["aliases"][0]["method"] == (
        "semantic_suffix_layer_occurrence"
    )


def test_reconstructs_requested_hf_name_from_native_metadata_name(tmp_path):
    native_shapes = OrderedDict(
        [
            ("module.sequential.0.word_embeddings.weight", (4, 2)),
            ("module.sequential.2.attention.dense.weight", (2, 2)),
            ("module.sequential.2.attention.dense.bias", (2,)),
            ("module.sequential.2.input_layernorm.weight", (2,)),
        ]
    )
    caller_shapes = OrderedDict(
        [
            ("gpt_neox.embed_in.weight", (4, 2)),
            ("gpt_neox.layers.0.attention.dense.weight", (2, 2)),
            ("gpt_neox.layers.0.attention.dense.bias", (2,)),
            ("gpt_neox.layers.0.input_layernorm.weight", (2,)),
        ]
    )
    torch.save(
        {"iteration": 143000, "param_shapes": [native_shapes]},
        tmp_path / "mp_rank_00_model_states.pt",
    )

    decay_m = torch.arange(1.0, 13.0)
    decay_v = decay_m.square()
    full_master = torch.arange(1.0, 17.0)
    for rank in range(2):
        payload = {
            "optimizer_state_dict": {
                "single_partition_of_fp32_groups": [
                    full_master[rank * 8 : (rank + 1) * 8].clone()
                ],
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": [
                                decay_m[rank * 6 : (rank + 1) * 6].clone()
                            ],
                            "exp_avg_sq": [
                                decay_v[rank * 6 : (rank + 1) * 6].clone()
                            ],
                        }
                    }
                },
            }
        }
        torch.save(
            payload,
            tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )

    requested = "gpt_neox.layers.0.attention.dense.weight"
    state = reconstruct_zero_adam_state(
        tmp_path,
        parameter_names=[requested],
        parameter_shapes=caller_shapes,
    )

    assert set(state.exp_avg) == {requested}
    assert torch.equal(state.exp_avg[requested], decay_m[8:12].reshape(2, 2))
    alignment = state.shape_validation["name_alignment"]
    assert alignment["alias_count"] == 1
    assert alignment["aliases"][0]["native_name"] == (
        "module.sequential.2.attention.dense.weight"
    )


def test_transposes_native_matrix_when_shape_contract_is_reversed():
    native = [OrderedDict([("module.sequential.2.mlp.dense_h_to_4h.weight", (3, 2))])]
    caller = OrderedDict(
        [("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", (2, 3))]
    )

    alignment = resolve_requested_parameter_names(caller.keys(), native, caller)

    assert alignment.transforms[
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight"
    ] == "transpose_2d"
