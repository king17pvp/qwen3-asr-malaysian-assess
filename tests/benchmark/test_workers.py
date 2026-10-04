"""Load-test clients spread over worker processes, so one event loop does not cap the load."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from asr_assess.benchmark.client import (
    LeastOutstandingTransport,
    OpenAITranscriptionTransport,
    make_transport,
)
from asr_assess.benchmark.load_stats import PoolClip
from asr_assess.benchmark.loadtest import ClientPlan, run_loadtest
from asr_assess.benchmark.workers import WorkerPool
from asr_assess.core.config import LoadTestConfig
from asr_assess.core.transcription_api import encode_wav
from tests.fakes import PidServer, QuickServer, fake_meta

WAV = encode_wav(np.zeros(3200, dtype=np.float32), 16000)
CLIPS = [PoolClip(f"c{i}", 0.2, "2-5", "a b", WAV) for i in range(5)]


def open_cfg(tmp: Path, levels: list[int]) -> LoadTestConfig:
    return LoadTestConfig.model_validate(
        {
            "profiles": {
                "t": {
                    "concurrency_levels": levels,
                    "repeats": 1,
                    "warmup_s": 0.1,
                    "steady_state_s": 0.6,
                    "bisect": False,
                    "bisect_max_steps": 0,
                    "stop_after_failures": 1,
                    "mode": "open",
                    "pause_s": 0.05,
                }
            },
            "request_timeout_s": 2,
            "gpu_sample_hz": 10,
            "metrics_scrape_hz": 1,
            "audio_pool": {"manifest": "x.jsonl", "seed": 1},
            "thresholds": {"p95_rtf_max": 0.5, "p95_rtf_strong": 0.3, "max_wer_delta_points": 1.0},
            "output_dir": str(tmp),
        }
    )


async def test_make_transport_balances_only_several_urls() -> None:
    one = make_transport("http://a:8000", 5.0, 256)
    two = make_transport("http://a:8000, http://a:8001", 5.0, 256)
    assert isinstance(one, OpenAITranscriptionTransport)
    assert isinstance(two, LeastOutstandingTransport)
    await one.aclose()
    await two.aclose()


def test_pool_needs_at_least_one_worker() -> None:
    with pytest.raises(ValueError, match="at least one"):
        WorkerPool(0, CLIPS, QuickServer)


async def test_pool_spreads_clients_over_processes_and_keeps_the_timetable() -> None:
    with WorkerPool(2, CLIPS, PidServer) as pool:
        start = time.monotonic() + 0.1
        plans = [
            ClientPlan(ids=(s, 4, 0), schedule=tuple((start + 0.1 * k, k) for k in range(3)))
            for s in range(4)
        ]
        records = await pool.run(plans)
    assert sorted(r.stream for r in records) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    pids = {r.text for r in records}
    assert len(pids) == 2 and str(os.getpid()) not in pids  # two workers, not this process
    assert all(r.due is not None and 0.0 <= r.sent - r.due < 0.05 for r in records)


async def test_pool_runs_closed_loop_streams_too() -> None:
    with WorkerPool(2, CLIPS, QuickServer) as pool:
        stop = time.monotonic() + 0.2
        plans = [ClientPlan(ids=(s, 2, 0), order=(0, 1, 2), stop_at=stop) for s in range(2)]
        records = await pool.run(plans)
    assert {r.stream for r in records} == {0, 1}
    assert all(r.status == "ok" and r.due is None for r in records)


async def test_run_loadtest_with_workers_measures_every_client(tmp_path: Path) -> None:
    with WorkerPool(2, CLIPS, QuickServer) as pool:
        config = open_cfg(tmp_path, [1, 4])
        out = await run_loadtest(
            QuickServer(), CLIPS, config, "t", tmp_path / "r", fake_meta(), None, workers=pool
        )
    rows = [
        json.loads(line) for line in (tmp_path / "r" / "requests.jsonl").read_text().splitlines()
    ]
    assert {row["stream"] for row in rows if row["level"] == 4} == {0, 1, 2, 3}
    assert all(row["status"] == "ok" for row in rows)
    assert out.client_workers == 2
    assert {v.threshold: v.max_sustainable for v in out.verdicts} == {0.5: 4, 0.3: 4}
