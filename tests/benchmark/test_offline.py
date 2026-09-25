"""Tests for single-stream RTF measurement."""

import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from asr_assess.benchmark.env_info import EnvInfo
from asr_assess.benchmark.offline import measure, run_offline, select_bench_set, summarize
from asr_assess.core.config import BenchConfig
from asr_assess.core.run_record import RunRecord
from asr_assess.inference.requests import to_requests
from tests.fakes import FakeClock, ScriptedEngine, write_clips

SPECS = [(f"s{i}", 3.0, "2-5", "x") for i in range(6)] + [
    (f"m{i}", 8.0, "5-15", "x") for i in range(2)
]


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


def env() -> EnvInfo:
    return EnvInfo([], None, None, None, None, "linux", "3.12", {})


def cfg(tmp_path: Path) -> BenchConfig:
    return BenchConfig(
        manifest=tmp_path / "eval.jsonl",
        clips_per_bucket=3,
        warmup_requests=2,
        repeats=2,
        seed=1,
        sample_rate=16000,
        output_dir=tmp_path / "results",
    )


class MemoryEngine(ScriptedEngine):
    def peak_memory_gb(self) -> float:
        return 5.5

    def reset_peak_memory(self) -> None:
        pass


def test_selection_is_capped_per_bucket_and_seeded(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    chosen = select_bench_set(entries, clips_per_bucket=3, seed=1)
    assert Counter(e.bucket for e in chosen) == {"2-5": 3, "5-15": 2}
    assert select_bench_set(entries, 3, seed=1) == chosen


def test_rtf_is_processing_over_audio_and_warmup_is_discarded(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS[:3])
    clock = FakeClock()
    engine = ScriptedEngine(clock=clock, rtf=0.25)
    timings = measure(
        engine, to_requests(entries, 16000, "auto"), entries, warmup=2, repeats=2, clock=clock
    )
    assert len(timings) == 6  # 3 clips x 2 repeats; the 2 warm-up calls are not recorded
    assert len(engine.calls) == 8
    assert all(t.rtf == pytest.approx(0.25) for t in timings)


def test_summary_groups_by_bucket(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    clock = FakeClock()
    timings = measure(ScriptedEngine(clock=clock, rtf=0.1), to_requests(entries, 16000, "auto"),
                      entries, warmup=0, repeats=1, clock=clock)  # fmt: skip
    summary = summarize(timings, peak_memory_gb=None)
    assert set(summary.by_bucket) == {"2-5", "5-15"}
    assert summary.overall.count == len(entries)
    assert summary.audio_s_per_s == pytest.approx(10.0)


def test_run_writes_results_with_peak_memory(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    clock = FakeClock()
    out_dir = tmp_path / "results" / "run"
    run_offline(
        MemoryEngine(clock=clock), entries, cfg(tmp_path), "auto", out_dir, record(), env(), clock
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["record"]["git_commit"] == "abc"
    assert summary["summary"]["peak_memory_gb"] == 5.5
    assert len((out_dir / "timings.jsonl").read_text(encoding="utf-8").splitlines()) == 10


def test_run_refuses_to_overwrite(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    out_dir = tmp_path / "results" / "run"
    run_offline(
        ScriptedEngine(), entries, cfg(tmp_path), "auto", out_dir, record(), env(), FakeClock()
    )
    with pytest.raises(FileExistsError):
        run_offline(
            ScriptedEngine(), entries, cfg(tmp_path), "auto", out_dir, record(), env(), FakeClock()
        )


def test_benchmark_does_not_depend_on_evaluation() -> None:
    code = (
        "import sys, asr_assess.benchmark.offline; "
        "sys.exit(any(m.startswith('asr_assess.evaluation') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
