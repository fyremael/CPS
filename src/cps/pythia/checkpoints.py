from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CheckpointDownload:
    repo_id: str
    revision: str
    local_dir: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def download_native_checkpoint(
    repo_id: str,
    revision: str,
    local_dir: str | Path,
    *,
    allow_patterns: Iterable[str] | None = None,
) -> CheckpointDownload:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'pythia' extra to download checkpoints") from exc
    target = Path(local_dir)
    patterns = list(allow_patterns) if allow_patterns is not None else [
        "*.yml",
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_*_optim_states.pt",
    ]
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=patterns,
    )
    files = tuple(str(path) for path in sorted(target.rglob("*")) if path.is_file())
    return CheckpointDownload(repo_id, revision, str(target), files)
