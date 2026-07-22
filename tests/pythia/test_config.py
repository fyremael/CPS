from pathlib import Path

import pytest

from cps.pythia.config import load_probe_config


def test_load_smoke_config():
    config = load_probe_config(Path("subjects/pythia/configs/pythia_70m_smoke.yaml"))
    assert config.model.run == "pythia-70m"
    assert config.basis.rank == 4
    assert isinstance(config.basis.components, tuple)


def test_unknown_config_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_probe_config(path)


def test_jacobian_and_attention_defaults_are_forward_ad_safe() -> None:
    from cps.pythia.config import JacobianConfig, ModelConfig, OutputConfig

    assert ModelConfig().attention_implementation == "eager"
    assert JacobianConfig().autodiff_backend == "auto"
    assert JacobianConfig().fallback_to_finite_difference is True
    assert OutputConfig().verbose is True
