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
