from collections import OrderedDict

import torch

from cps.pythia.native_state import reconstruct_zero_adam_state


def test_reconstruct_partitioned_adam_state(tmp_path):
    shapes = [OrderedDict([("a", (3,)), ("b", (2,))])]
    torch.save({"param_shapes": shapes}, tmp_path / "mp_rank_00_model_states.pt")
    full_m = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    full_v = full_m.square()
    for rank in range(2):
        sl = slice(rank * 3, (rank + 1) * 3)
        payload = {
            "optimizer_state_dict": {
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": full_m[sl].clone(),
                            "exp_avg_sq": full_v[sl].clone(),
                        }
                    }
                }
            }
        }
        torch.save(payload, tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    state = reconstruct_zero_adam_state(tmp_path)
    assert torch.equal(state.exp_avg["a"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(state.exp_avg["b"], torch.tensor([4.0, 5.0]))
    assert state.partition_count == 2



def _save_partitioned_payloads(tmp_path, *, full_m, full_v, full_w):
    for rank in range(2):
        sl = slice(rank * 3, (rank + 1) * 3)
        payload = {
            "optimizer_state_dict": {
                "single_partition_of_fp32_groups": [full_w[sl].clone()],
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": full_m[sl].clone(),
                            "exp_avg_sq": full_v[sl].clone(),
                        }
                    }
                },
            }
        }
        torch.save(payload, tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")


def test_reconstructs_when_old_metadata_omits_param_shapes(tmp_path):
    torch.save({"iteration": 143000}, tmp_path / "mp_rank_00_model_states.pt")
    full_m = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    full_v = full_m.square()
    full_w = torch.tensor([0.5, -1.0, 2.0, 3.0, -4.0, 0.0])
    _save_partitioned_payloads(tmp_path, full_m=full_m, full_v=full_v, full_w=full_w)
    shapes = OrderedDict([("a", (3,)), ("b", (2,))])
    references = {
        "a": {"l2": float(full_w[:3].norm()), "l1": float(full_w[:3].abs().sum())},
        "b": {"l2": float(full_w[3:5].norm()), "l1": float(full_w[3:5].abs().sum())},
    }

    state = reconstruct_zero_adam_state(
        tmp_path,
        parameter_shapes=shapes,
        reference_signatures=references,
    )

    assert state.shape_source == "validated_model_parameter_order"
    assert state.shape_validation["match_fraction"] == 1.0
    assert torch.equal(state.exp_avg["a"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(state.exp_avg["b"], torch.tensor([4.0, 5.0]))


def test_rejects_unvalidated_model_parameter_order(tmp_path):
    import pytest

    torch.save({"iteration": 143000}, tmp_path / "mp_rank_00_model_states.pt")
    full_m = torch.arange(1.0, 7.0)
    full_v = full_m.square()
    full_w = torch.tensor([0.5, -1.0, 2.0, 3.0, -4.0, 0.0])
    _save_partitioned_payloads(tmp_path, full_m=full_m, full_v=full_v, full_w=full_w)
    shapes = OrderedDict([("a", (3,)), ("b", (2,))])
    wrong = {
        "a": {"l2": 100.0, "l1": 100.0},
        "b": {"l2": 100.0, "l1": 100.0},
    }

    with pytest.raises(ValueError, match="failed validation"):
        reconstruct_zero_adam_state(
            tmp_path,
            parameter_shapes=shapes,
            reference_signatures=wrong,
        )


def test_finds_param_shapes_in_optimizer_metadata(tmp_path):
    shapes = [OrderedDict([("a", (3,)), ("b", (2,))])]
    torch.save({"iteration": 1}, tmp_path / "mp_rank_00_model_states.pt")
    full_m = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    full_v = full_m.square()
    for rank in range(2):
        sl = slice(rank * 3, (rank + 1) * 3)
        payload = {
            "param_shapes": shapes,
            "optimizer_state_dict": {
                "optimizer_state_dict": {
                    "state": {
                        0: {
                            "exp_avg": full_m[sl].clone(),
                            "exp_avg_sq": full_v[sl].clone(),
                        }
                    }
                }
            },
        }
        torch.save(payload, tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")

    state = reconstruct_zero_adam_state(tmp_path)
    assert state.shape_source == "optimizer_checkpoint_param_shapes"
    assert torch.equal(state.exp_avg["b"], torch.tensor([4.0, 5.0]))
