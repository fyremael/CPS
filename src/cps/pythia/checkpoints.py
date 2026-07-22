from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_FINAL_PYTHIA_STEP = 143_000
_STEP_PATTERN = re.compile(r"^step(?P<step>\d+)$")


class NativeCheckpointRevisionError(ValueError):
    """Raised when a requested native checkpoint revision is not exposed by the Hub repo."""


@dataclass(frozen=True)
class CheckpointDownload:
    repo_id: str
    revision: str
    local_dir: str
    files: tuple[str, ...]
    requested_revision: str | None = None
    training_step: int | None = None
    available_branches: tuple[str, ...] = ()
    available_tags: tuple[str, ...] = ()
    resolution_note: str | None = None

    @property
    def resolved_revision(self) -> str:
        """Explicit alias for the revision that was actually downloaded."""

        return self.revision

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resolved_revision"] = self.revision
        return payload


def _ref_names(refs: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    branches = tuple(sorted(str(item.name) for item in getattr(refs, "branches", ())))
    tags = tuple(sorted(str(item.name) for item in getattr(refs, "tags", ())))
    return branches, tags


def _list_repo_refs(api: Any, repo_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return _ref_names(api.list_repo_refs(repo_id))
    except Exception:
        # Ref discovery is diagnostic. A commit hash or transient Hub API failure should not
        # prevent snapshot_download from attempting the caller's requested revision.
        return (), ()


def _is_pythia_native_repo(repo_id: str) -> bool:
    return repo_id.startswith("EleutherAI/neox-ckpt-pythia-")


def _training_step(repo_id: str, requested: str, resolved: str) -> int | None:
    for candidate in (requested, resolved):
        match = _STEP_PATTERN.fullmatch(candidate)
        if match is not None:
            return int(match.group("step"))
    if _is_pythia_native_repo(repo_id) and resolved == "main":
        return _FINAL_PYTHIA_STEP
    return None


def _resolve_revision(
    repo_id: str,
    requested: str,
    branches: tuple[str, ...],
    tags: tuple[str, ...],
) -> tuple[str, str | None]:
    available = set(branches) | set(tags)
    if requested in available or not available:
        return requested, None

    # The public Pythia weight repositories expose step143000 as a branch, but current
    # native GPT-NeoX optimizer-state repositories may expose the final checkpoint only on
    # main. Prefer an actual step143000 ref when it exists; otherwise resolve that semantic
    # final-step request to main and record the substitution explicitly.
    if (
        requested == f"step{_FINAL_PYTHIA_STEP}"
        and _is_pythia_native_repo(repo_id)
        and "main" in available
    ):
        return (
            "main",
            f"Requested {requested}, but {repo_id} does not expose that ref; "
            "resolved the final native Pythia checkpoint to main.",
        )

    return requested, None


def _revision_error(
    repo_id: str,
    requested: str,
    branches: tuple[str, ...],
    tags: tuple[str, ...],
) -> NativeCheckpointRevisionError:
    available = tuple(sorted(set(branches) | set(tags)))
    rendered = ", ".join(available) if available else "unavailable (Hub ref discovery failed)"
    guidance = ""
    if _is_pythia_native_repo(repo_id):
        guidance = (
            " Current Pythia native repositories may expose only the final optimizer state "
            "on 'main'. Historical Transformer weights and historical native optimizer moments "
            "are distinct artifacts; do not substitute weights for missing moments."
        )
    return NativeCheckpointRevisionError(
        f"Native checkpoint revision {requested!r} is not available in {repo_id!r}. "
        f"Available refs: {rendered}.{guidance}"
    )


def download_native_checkpoint(
    repo_id: str,
    revision: str,
    local_dir: str | Path,
    *,
    allow_patterns: Iterable[str] | None = None,
) -> CheckpointDownload:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import RevisionNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra to download checkpoints") from exc

    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    patterns = list(allow_patterns) if allow_patterns is not None else [
        "*.yml",
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_*_optim_states.pt",
    ]

    api = HfApi()
    branches, tags = _list_repo_refs(api, repo_id)
    resolved_revision, resolution_note = _resolve_revision(repo_id, revision, branches, tags)

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            local_dir=target,
            allow_patterns=patterns,
        )
    except RevisionNotFoundError as exc:
        # Refresh refs once so the error reports the current Hub state rather than a stale
        # preflight result.
        branches, tags = _list_repo_refs(api, repo_id)
        raise _revision_error(repo_id, revision, branches, tags) from exc

    remote_files = tuple(
        str(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and ".cache" not in path.parts and path.name != "cps_download_manifest.json"
    )
    result = CheckpointDownload(
        repo_id=repo_id,
        revision=resolved_revision,
        local_dir=str(target),
        files=remote_files,
        requested_revision=revision,
        training_step=_training_step(repo_id, revision, resolved_revision),
        available_branches=branches,
        available_tags=tags,
        resolution_note=resolution_note,
    )
    (target / "cps_download_manifest.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = tuple(
        str(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    )
    return CheckpointDownload(
        repo_id=result.repo_id,
        revision=result.revision,
        local_dir=result.local_dir,
        files=files,
        requested_revision=result.requested_revision,
        training_step=result.training_step,
        available_branches=result.available_branches,
        available_tags=result.available_tags,
        resolution_note=result.resolution_note,
    )
