"""Tests for running an engine over a manifest and writing results."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from asr_assess.core.config import BootstrapSettings, EvalConfig
from asr_assess.core.run_record import RunRecord
from asr_assess.evaluation.evaluate import evaluate, run_batches, to_requests
from tests.fakes import ScriptedEngine, write_clips

SPECS = [("a", 3.0, "2-5", "satu dua"), ("b", 6.0, "5-15", "tiga empat"), ("c", 2.5, "2-5", "lima")]


def cfg(tmp_path: Path, batch_size: int = 1) -> EvalConfig:
    return EvalConfig(
        manifests={"eval": tmp_path / "eval.jsonl"},
        batch_size=batch_size,
        sample_rate=16000,
        bootstrap=BootstrapSettings(n_resamples=100, confidence=0.95, seed=0),
        output_dir=tmp_path / "results",
    )


def record() -> RunRecord:
    return RunRecord(
        git_commit="abc",
        git_dirty=False,
        config={},
        config_hash="h",
        gpu_name=None,
        python_version="3.12",
        package_versions={},
        timestamp=datetime(2026, 9, 26, tzinfo=UTC),
    )


def test_requests_carry_manifest_language_only_when_asked(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    assert {r.language for r in to_requests(entries, 16000, "auto")} == {None}
    assert {r.language for r in to_requests(entries, 16000, "manifest")} == {"Malay"}


def test_missing_audio_names_the_file(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    Path(entries[0].audio).unlink()
    with pytest.raises(FileNotFoundError, match=r"a\.wav"):
        to_requests(entries, 16000, "auto")


def test_batches_are_capped_and_order_is_kept(tmp_path: Path) -> None:
    engine = ScriptedEngine()
    requests = to_requests(write_clips(tmp_path, SPECS), 16000, "auto")
    out = run_batches(engine, requests, batch_size=2)
    assert [len(call) for call in engine.calls] == [2, 1]
    assert [t.id for t in out] == [r.id for r in requests]


def test_perfect_engine_scores_zero_and_writes_results(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    engine = ScriptedEngine({e.audio: e.transcript for e in entries})
    out_dir = tmp_path / "results" / "run"

    report = evaluate(engine, entries, cfg(tmp_path), "auto", out_dir, record())

    assert report.overall.wer.value == 0.0
    lines = (out_dir / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["record"]["git_commit"] == "abc"
    assert metrics["engine"] == "scripted"
    assert metrics["truncated"] == 0
    assert metrics["report"]["overall"]["n"] == 3


def test_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    out_dir = tmp_path / "results" / "run"
    evaluate(ScriptedEngine(), entries, cfg(tmp_path), "auto", out_dir, record())
    with pytest.raises(FileExistsError):
        evaluate(ScriptedEngine(), entries, cfg(tmp_path), "auto", out_dir, record())


def test_hypotheses_keep_raw_text_next_to_normalized(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS[:1])
    engine = ScriptedEngine({entries[0].audio: "Satu, DUA!"})
    out_dir = tmp_path / "results" / "run"
    evaluate(engine, entries, cfg(tmp_path), "auto", out_dir, record())
    row = json.loads((out_dir / "hypotheses.jsonl").read_text(encoding="utf-8"))
    assert row["hypothesis_raw"] == "Satu, DUA!"
    assert row["hypothesis"] == "satu dua"
    assert row["reference_raw"] == "satu dua"
