from cps.plot_labels import compact_basis_labels, compact_parameter_name


def test_pythia_block_name_is_compacted_semantically():
    assert (
        compact_parameter_name("gpt_neox.layers.0.attention.dense.weight")
        == "L0.attn.out"
    )


def test_shared_block_uses_component_only_axis_labels():
    basis = [
        {"name": "gpt_neox.layers.0.attention.dense.weight:theta"},
        {"name": "gpt_neox.layers.0.attention.dense.weight:update"},
        {"name": "gpt_neox.layers.0.attention.dense.weight:random0"},
        {"name": "gpt_neox.layers.0.attention.dense.weight:random1"},
    ]
    labels, mapping, shared_block, key = compact_basis_labels(basis)
    assert labels == ["θ", "Δθ", "r₀", "r₁"]
    assert shared_block == "L0.attn.out"
    assert mapping[basis[1]["name"]] == "Δθ"
    assert key[0] == ("θ", basis[0]["name"])


def test_mixed_blocks_keep_compact_block_identity():
    basis = [
        {"name": "gpt_neox.layers.0.attention.dense.weight:theta"},
        {"name": "gpt_neox.layers.1.mlp.dense_4h_to_h.weight:update"},
    ]
    labels, _, shared_block, _ = compact_basis_labels(basis)
    assert labels == ["L0.attn.out · θ", "L1.mlp.down · Δθ"]
    assert shared_block is None
