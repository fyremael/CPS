from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXCLUDES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
}


def included(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    return not bool(parts & EXCLUDES) and path.suffix not in {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DIST.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    release_name = f"coupling-phase-spectroscopy-{stamp}"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / release_name
        stage.mkdir()
        for source in sorted(ROOT.rglob("*")):
            if not source.is_file() or not included(source):
                continue
            target = stage / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        archive = Path(shutil.make_archive(str(DIST / release_name), "zip", stage.parent, stage.name))

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "archive": archive.name,
        "sha256": sha256(archive),
        "git_commit": _git_commit(),
    }
    manifest_path = DIST / f"{release_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(archive)
    print(manifest_path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
