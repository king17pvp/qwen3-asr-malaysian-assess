"""Provenance (git commit, config, GPU, versions, time) stamped on every result file."""

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class RunRecord(BaseModel):
    """Everything needed to trace a result back to its code, config and hardware."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_commit: str | None
    git_dirty: bool
    config: dict[str, Any]
    config_hash: str
    gpu_name: str | None
    python_version: str
    package_versions: dict[str, str | None]
    timestamp: datetime


def config_hash(config: BaseModel) -> str:
    """SHA-256 of the config's canonical JSON form."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_state(repo: Path) -> tuple[str | None, bool]:
    """HEAD commit and whether tracked files have changes; (None, False) outside a repo.

    Untracked files (notes, data) are ignored: they cannot change what the committed code does.
    """
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if commit is None:
        return None, False
    status = _run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    return commit, bool(status)


def gpu_name() -> str | None:
    """Name of the first NVIDIA GPU, or None when no GPU or driver is present."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Installed version of each distribution, or None if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_run_record(
    config: BaseModel, packages: Iterable[str], repo: Path, now: datetime | None = None
) -> RunRecord:
    """Snapshot provenance for a run of ``config``."""
    commit, dirty = git_state(repo)
    return RunRecord(
        git_commit=commit,
        git_dirty=dirty,
        config=config.model_dump(mode="json"),
        config_hash=config_hash(config),
        gpu_name=gpu_name(),
        python_version=platform.python_version(),
        package_versions=package_versions(packages),
        timestamp=now or datetime.now(UTC),
    )


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()
