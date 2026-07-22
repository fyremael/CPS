from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cps.pythia.checkpoints import (
    NativeCheckpointRevisionError,
    download_native_checkpoint,
)


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    branches: tuple[str, ...],
    tags: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> list[str]:
    calls: list[str] = []

    class RevisionNotFoundError(Exception):
        pass

    class HfApi:
        def list_repo_refs(self, repo_id: str):
            assert repo_id == "EleutherAI/neox-ckpt-pythia-70m"
            return SimpleNamespace(
                branches=[SimpleNamespace(name=name) for name in branches],
                tags=[SimpleNamespace(name=name) for name in tags],
            )

    def snapshot_download(*, repo_id, revision, local_dir, allow_patterns):
        assert repo_id == "EleutherAI/neox-ckpt-pythia-70m"
        assert "zero_pp_rank_*_optim_states.pt" in allow_patterns
        calls.append(revision)
        if revision in missing:
            raise RevisionNotFoundError(revision)
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "mp_rank_00_model_states.pt").write_bytes(b"metadata")
        (target / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"moments")
        return str(target)

    hub = ModuleType("huggingface_hub")
    hub.HfApi = HfApi
    hub.snapshot_download = snapshot_download
    errors = ModuleType("huggingface_hub.errors")
    errors.RevisionNotFoundError = RevisionNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return calls


def test_final_step_resolves_to_main_when_native_ref_is_absent(tmp_path, monkeypatch):
    calls = _install_fake_hub(monkeypatch, branches=("main",))

    result = download_native_checkpoint(
        "EleutherAI/neox-ckpt-pythia-70m",
        "step143000",
        tmp_path,
    )

    assert calls == ["main"]
    assert result.requested_revision == "step143000"
    assert result.resolved_revision == "main"
    assert result.training_step == 143000
    assert result.resolution_note is not None
    manifest = json.loads((tmp_path / "cps_download_manifest.json").read_text())
    assert manifest["requested_revision"] == "step143000"
    assert manifest["resolved_revision"] == "main"


def test_actual_step_ref_is_preferred_when_exposed(tmp_path, monkeypatch):
    calls = _install_fake_hub(monkeypatch, branches=("main", "step143000"))

    result = download_native_checkpoint(
        "EleutherAI/neox-ckpt-pythia-70m",
        "step143000",
        tmp_path,
    )

    assert calls == ["step143000"]
    assert result.resolved_revision == "step143000"
    assert result.resolution_note is None


def test_missing_intermediate_native_revision_fails_closed(tmp_path, monkeypatch):
    _install_fake_hub(
        monkeypatch,
        branches=("main",),
        missing=("step1000",),
    )

    with pytest.raises(NativeCheckpointRevisionError) as error:
        download_native_checkpoint(
            "EleutherAI/neox-ckpt-pythia-70m",
            "step1000",
            tmp_path,
        )

    message = str(error.value)
    assert "step1000" in message
    assert "Available refs: main" in message
    assert "do not substitute weights for missing moments" in message
