import torch

from cps.pythia.blocks import build_parameter_blocks, select_parameter_names


def test_semantic_block_classification_and_selection():
    params = {
        "gpt_neox.layers.0.attention.dense.weight": torch.zeros(2, 2),
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight": torch.zeros(4, 2),
    }
    names = select_parameter_names(params, ["*attention*dense*"])
    assert names == ("gpt_neox.layers.0.attention.dense.weight",)
    blocks = build_parameter_blocks(params.items())
    assert blocks[0].layer == 0
    assert blocks[0].subsystem == "attention"
    assert blocks[1].subsystem == "mlp"
