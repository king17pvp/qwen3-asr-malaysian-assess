"""Tests for the `asr-assess` CLI entry point."""

import json
import subprocess
import sys
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner, Result

from asr_assess.cli import app
from asr_assess.core.manifest import write_manifest
from tests.fakes import ScriptedEngine, write_clips

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "check-config" in click.unstyle(result.output)


def test_check_config_accepts_shipped_file() -> None:
    result = runner.invoke(app, ["check-config", "lora", str(CONFIGS / "lora.yaml")])
    assert result.exit_code == 0, result.output


def test_check_config_accepts_data_config() -> None:
    result = runner.invoke(app, ["check-config", "data", str(CONFIGS / "data.yaml")])
    assert result.exit_code == 0, result.output


def test_data_command_offers_dry_run() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in click.unstyle(result.output)


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


def test_check_config_accepts_inference_configs() -> None:
    for kind, path in [
        ("engine", "engines/hf_base.yaml"),
        ("eval", "eval.yaml"),
        ("bench", "bench.yaml"),
        ("engine", "engines/hf_ft.yaml"),
    ]:
        result = runner.invoke(app, ["check-config", kind, str(CONFIGS / path)])
        assert result.exit_code == 0, result.output


ENGINE = str(CONFIGS / "engines" / "hf_base.yaml")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp repo root with a tiny manifest, configs pointing at it, and a scripted engine."""
    entries = write_clips(tmp_path, [("a", 3.0, "2-5", "satu dua"), ("b", 6.0, "5-15", "tiga")])
    write_manifest(tmp_path / "eval.jsonl", entries)
    (tmp_path / "eval.yaml").write_text(
        f"manifests: {{eval: {tmp_path / 'eval.jsonl'}}}\nbatch_size: 1\nsample_rate: 16000\n"
        "bootstrap: {n_resamples: 50, confidence: 0.95, seed: 0}\n"
        f"output_dir: {tmp_path / 'res'}\n",
        encoding="utf-8",
    )
    (tmp_path / "bench.yaml").write_text(
        f"manifest: {tmp_path / 'eval.jsonl'}\nclips_per_bucket: 5\nwarmup_requests: 1\n"
        f"repeats: 1\nseed: 0\nsample_rate: 16000\noutput_dir: {tmp_path / 'res'}\n",
        encoding="utf-8",
    )
    texts = {e.audio: e.transcript for e in entries}
    monkeypatch.setattr(
        "asr_assess.inference.factory.make_engine", lambda cfg: ScriptedEngine(texts)
    )
    return tmp_path


def run_eval(workspace: Path, *extra: str) -> Result:
    args = ["eval", "--engine", ENGINE, "--config", str(workspace / "eval.yaml"), *extra]
    return runner.invoke(app, args)


def test_eval_command_writes_metrics(workspace: Path) -> None:
    result = run_eval(workspace, "--run-name", "r1")
    assert result.exit_code == 0, result.output
    metrics = json.loads((workspace / "res" / "r1" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["report"]["overall"]["wer"]["value"] == 0.0
    assert metrics["record"]["config"]["manifest"] == "eval"


def test_eval_limit_takes_a_tiny_sample(workspace: Path) -> None:
    result = run_eval(workspace, "--limit", "1", "--run-name", "r2")
    assert result.exit_code == 0, result.output
    metrics = json.loads((workspace / "res" / "r2" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n"] == 1


def test_eval_rejects_unknown_manifest_name(workspace: Path) -> None:
    result = run_eval(workspace, "--manifest", "nope")
    assert result.exit_code == 1, result.output


def test_bench_command_writes_summary(workspace: Path) -> None:
    config = str(workspace / "bench.yaml")
    args = ["bench", "--engine", ENGINE, "--config", config, "--run-name", "b1"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (workspace / "res" / "b1" / "summary.json").exists()
