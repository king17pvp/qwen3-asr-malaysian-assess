"""Tests for the `asr-assess` CLI entry point."""

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from asr_assess.cli import app

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "check-config" in result.output


def test_check_config_accepts_shipped_file() -> None:
    result = runner.invoke(app, ["check-config", "lora", str(CONFIGS / "lora.yaml")])
    assert result.exit_code == 0, result.output


def test_check_config_rejects_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("rank: 16\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config", "lora", str(bad)])
    assert result.exit_code != 0


def test_import_does_not_load_heavy_dependencies() -> None:
    code = (
        "import sys, asr_assess.cli; "
        "heavy = {'torch', 'transformers', 'peft', 'vllm', 'librosa'} & set(sys.modules); "
        "sys.exit(len(heavy))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
