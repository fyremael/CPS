import pytest

from cps.pythia.registry import PYTHIA_STEPS, get_run_spec, list_run_specs


def test_checkpoint_grid_has_154_steps():
    assert len(PYTHIA_STEPS) == 154
    assert PYTHIA_STEPS[:11] == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert PYTHIA_STEPS[-1] == 143000


def test_registry_resolves_pythia_and_polypythia():
    assert get_run_spec("pythia-70m").model_id == "EleutherAI/pythia-70m"
    assert get_run_spec("polypythia-70m-seed7").seed == 7
    assert len(list_run_specs()) >= 20


def test_revision_rejects_noncanonical_step():
    with pytest.raises(ValueError):
        get_run_spec("pythia-70m").revision(999)
