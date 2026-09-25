"""Tests for the provenance record stamped on every result file."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from asr_assess.core.config import StrictModel
from asr_assess.core.run_record import (
    collect_run_record,
    config_hash,
    git_state,
    package_versions,
)


class Cfg(StrictModel):
    lr: float
    name: str


def init_repo(path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")


class TestConfigHash:
    def test_is_stable_for_equal_configs(self) -> None:
        assert config_hash(Cfg(lr=1e-4, name="a")) == config_hash(Cfg(lr=1e-4, name="a"))

    def test_changes_with_any_value(self) -> None:
        assert config_hash(Cfg(lr=1e-4, name="a")) != config_hash(Cfg(lr=2e-4, name="a"))


class TestGitState:
    def test_clean_repo(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        commit, dirty = git_state(tmp_path)
        assert commit is not None and len(commit) == 40
        assert dirty is False

    def test_untracked_file_marks_dirty(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        (tmp_path / "new.txt").write_text("x", encoding="utf-8")
        assert git_state(tmp_path)[1] is True

    def test_outside_a_repo(self, tmp_path: Path) -> None:
        assert git_state(tmp_path) == (None, False)


class TestPackageVersions:
    def test_installed_and_missing(self) -> None:
        versions = package_versions(["numpy", "surely-not-installed-pkg"])
        assert versions["numpy"]
        assert versions["surely-not-installed-pkg"] is None


class TestCollect:
    def test_stamps_config_git_and_time(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        cfg = Cfg(lr=1e-4, name="a")
        now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)

        record = collect_run_record(cfg, packages=["numpy"], repo=tmp_path, now=now)

        assert record.config_hash == config_hash(cfg)
        assert record.config == {"lr": 1e-4, "name": "a"}
        assert record.git_commit is not None
        assert record.timestamp == now
        assert "numpy" in record.package_versions
