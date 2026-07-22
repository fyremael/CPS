from __future__ import annotations

import json
from pathlib import Path


def patch_runner() -> None:
    path = Path("src/cps/pythia/runner.py")
    text = path.read_text(encoding="utf-8")
    text = text.replace("import json\n", "import json\nfrom collections import OrderedDict\n", 1)
    text = text.replace(
        "from .native_state import reconstruct_zero_adam_state\n",
        "from .native_state import parameter_reference_signatures, reconstruct_zero_adam_state\n",
        1,
    )
    old = """        native = reconstruct_zero_adam_state(
            config.state.native_checkpoint_dir,
            parameter_names=selected_names,
        )
"""
    new = """        reporter.info(
            "Preparing the full ordered model-shape contract. Historical GPT-NeoX packets may "
            "omit param_shapes; CPS will use this contract only after validating it against the "
            "native fp32 master-weight partitions."
        )
        ordered_shapes = OrderedDict(
            (name, tuple(parameter.shape)) for name, parameter in model.named_parameters()
        )
        signatures = parameter_reference_signatures(all_named)
        native = reconstruct_zero_adam_state(
            config.state.native_checkpoint_dir,
            parameter_names=selected_names,
            parameter_shapes=ordered_shapes,
            reference_signatures=signatures,
        )
"""
    if old not in text:
        raise RuntimeError("native runner call not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        '                "parameter_count": native.parameter_count,\n',
        '                "parameter_count": native.parameter_count,\n'
        '                "shape_source": native.shape_source,\n'
        '                "shape_validation": native.shape_validation,\n',
        1,
    )
    text = text.replace(
        '        reporter.metric("reconstructed moment parameters", native.parameter_count)\n',
        '        reporter.metric("reconstructed moment parameters", native.parameter_count)\n'
        '        reporter.metric("native shape source", native.shape_source)\n'
        '        if native.shape_validation.get("match_fraction") is not None:\n'
        '            reporter.metric(\n'
        '                "native order signature match",\n'
        '                f"{native.shape_validation[\'match_fraction\']:.1%}",\n'
        '            )\n',
        1,
    )
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    path = Path("tests/pythia/test_native_state.py")
    text = path.read_text(encoding="utf-8")
    marker = "def test_reconstructs_when_old_metadata_omits_param_shapes"
    if marker in text:
        return
    text += '''


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
'''
    path.write_text(text, encoding="utf-8")


def patch_notebook() -> None:
    path = Path("notebooks/04_native_optimizer_state.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "output = run_probe(config)" in source:
            cell["source"] = [
                "from cps.pythia.runner import run_probe\n",
                "\n",
                'print("[NATIVE] Reconstructing Adam moments from ZeRO partitions.", flush=True)\n',
                'print("[NATIVE] If param_shapes are absent, CPS will test the Transformers parameter order", flush=True)\n',
                'print("[NATIVE] against native fp32 master-weight tensor signatures before assigning names.", flush=True)\n',
                "output = run_probe(config)\n",
                'print(f"[NATIVE] evidence root={output}", flush=True)\n',
            ]
        if "## Stage 2 — bind the native moments" in source:
            cell["source"] = [
                "## Stage 2 — bind and validate the native moments\n",
                "\n",
                "The optimizer step number is derived from the semantic checkpoint revision. Historical GPT-NeoX pipeline metadata may omit `param_shapes`. CPS first searches model and optimizer metadata; if both omit the field, it proposes the loaded Transformers parameter order and accepts it only when tensorwise L1/L2 signatures agree with the native fp32 master-weight partitions. The evidence manifest records the shape source and validation rate.\n",
            ]
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def patch_version() -> None:
    path = Path("pyproject.toml")
    text = path.read_text(encoding="utf-8")
    text = text.replace('version = "0.2.5"', 'version = "0.2.6"', 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_runner()
    patch_tests()
    patch_notebook()
    patch_version()
    for temporary in (
        Path("scripts/apply_native_shape_fallback_patch.py"),
        Path(".github/workflows/apply-native-shape-fallback-pr.yml"),
    ):
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
