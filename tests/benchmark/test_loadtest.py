"""End-to-end load-test driver tests against a fake server whose latency grows with load."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from asr_assess.benchmark.load_stats import PoolClip
from asr_assess.benchmark.loadtest import build_pool, run_loadtest, run_session
from asr_assess.core.config import LoadTestConfig
from asr_assess.core.transcription_api import encode_wav
from tests.fakes import FakeTransport, fake_meta, write_clips

AUDIO_S = 0.2
WAV = encode_wav(np.zeros(3200, dtype=np.float32), 16000)
CLIPS = [PoolClip(f"c{i}", AUDIO_S, "2-5", "a b", WAV) for i in range(5)]


def cfg(tmp: Path, levels: list[int], **profile: object) -> LoadTestConfig:
    return LoadTestConfig.model_validate(
        {
            "profiles": {
                "t": {
                    "concurrency_levels": levels,
                    "repeats": 1,
                    "warmup_s": 0.1,
                    "steady_state_s": 0.6,
                    "bisect": True,
                    "bisect_max_steps": 3,
                    "stop_after_failures": 1,
                    **profile,
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


def verdicts(summary_verdicts: list) -> dict[float, int | None]:  # type: ignore[type-arg]
    return {v.threshold: v.max_sustainable for v in summary_verdicts}


async def test_finds_the_limit_with_bisection(tmp_path: Path) -> None:
    # latency = 0.018 s x in-flight -> RTF 0.09 x level: 5 passes (0.45), 6 fails (0.54)
    fake = FakeTransport(latency=lambda n: 0.018 * n, text=lambda wav, n: "a b")
    out = await run_loadtest(
        fake, CLIPS, cfg(tmp_path, [1, 2, 4, 8, 16]), "t", tmp_path / "run", fake_meta(), gpu=None
    )
    run_dir = tmp_path / "run"
    ran = [
        json.loads(line)["level"] for line in (run_dir / "levels.jsonl").read_text().splitlines()
    ]
    assert ran == [1, 2, 4, 8, 16, 6, 5]  # one level past the first failure, then bisection
    assert [lv.level for lv in out.levels] == sorted(ran)  # the summary is ordered by level
    assert fake.max_inflight == 16
    # bisection searches only the 0.5 boundary; level 3 is never run, so the 0.3 verdict is
    # the highest tested passing level: 2 (RTF 0.18; 4 gives 0.36)
    assert verdicts(out.verdicts) == {0.5: 5, 0.3: 2}
    rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    assert rows and all(row["status"] == "ok" for row in rows)
    assert json.loads((run_dir / "summary.json").read_text())["label"] == "t"


async def test_wer_under_load_can_fail_a_level(tmp_path: Path) -> None:
    fake = FakeTransport(latency=lambda n: 0.001, text=lambda wav, n: "a b" if n < 4 else "x y")
    out = await run_loadtest(
        fake, CLIPS, cfg(tmp_path, [1, 2, 4]), "t", tmp_path / "r", fake_meta(), gpu=None
    )
    assert out.reference_wer == 0.0
    assert verdicts(out.verdicts)[0.5] == 3  # 4 fails on WER; bisection finds 3


async def test_failed_requests_stop_the_sweep(tmp_path: Path) -> None:
    fake = FakeTransport(latency=lambda n: 0.001, text=lambda wav, n: "a b", fail_at=2)
    out = await run_loadtest(
        fake, CLIPS, cfg(tmp_path, [1, 2, 4, 8, 16]), "t", tmp_path / "r", fake_meta(), gpu=None
    )
    assert [lv.level for lv in out.levels] == [1, 2, 4]
    assert out.levels[1].n_error > 0
    assert verdicts(out.verdicts)[0.5] == 1


async def test_refuses_existing_label(tmp_path: Path) -> None:
    (tmp_path / "r").mkdir()
    with pytest.raises(FileExistsError):
        await run_loadtest(
            FakeTransport(lambda n: 0.0),
            CLIPS,
            cfg(tmp_path, [1]),
            "t",
            tmp_path / "r",
            fake_meta(),
            gpu=None,
        )


async def test_unready_server_fails_fast(tmp_path: Path) -> None:
    class Down(FakeTransport):
        async def wait_ready(self, timeout_s: float) -> bool:
            return False

    with pytest.raises(RuntimeError, match="not ready"):
        await run_loadtest(
            Down(lambda n: 0.0), CLIPS, cfg(tmp_path, [1]), "t", tmp_path / "r", fake_meta(), None
        )
    assert not (tmp_path / "r").exists()


async def test_every_request_sends_unique_audio(tmp_path: Path) -> None:
    sent: list[bytes] = []

    def text(wav: bytes, n: int) -> str:
        sent.append(wav)
        return "a b"

    fake = FakeTransport(latency=lambda n: 0.001, text=text)
    await run_loadtest(fake, CLIPS, cfg(tmp_path, [1, 2]), "t", tmp_path / "r", fake_meta(), None)
    assert len(sent) > 2 * len(CLIPS)  # every clip was sent more than once
    assert len(set(sent)) == len(sent)


async def test_open_session_sends_on_the_speaking_schedule_without_waiting() -> None:
    # the server takes 0.3 s; the speaker sends every 0.1 s regardless
    fake = FakeTransport(latency=lambda n: 0.3, text=lambda wav, n: "a b")
    t0 = time.monotonic() + 0.05
    schedule = [(t0 + 0.1 * k, k % len(CLIPS)) for k in range(5)]
    records = await run_session(fake, CLIPS, schedule, (0, 1, 0), time.monotonic)
    assert [r.id for r in records] == [f"c{k % len(CLIPS)}" for k in range(5)]
    assert all(abs(r.sent - at) < 0.03 for r, (at, _) in zip(records, schedule, strict=True))
    assert fake.max_inflight >= 3
    assert all(r.status == "ok" and r.finished - r.sent >= 0.29 for r in records)
    assert [r.due for r in records] == [at for at, _ in schedule]
    assert all(r.due is not None and r.sent >= r.due for r in records)


async def test_open_run_measures_every_level_and_records_its_mode(tmp_path: Path) -> None:
    fake = FakeTransport(latency=lambda n: 0.01, text=lambda wav, n: "a b")
    config = cfg(tmp_path, [1, 4, 8], mode="open", pause_s=0.05)
    out = await run_loadtest(fake, CLIPS, config, "t", tmp_path / "run", fake_meta(), None)
    assert [lv.level for lv in out.levels] == [1, 4, 8]
    assert all(lv.n_ok == lv.n_sent > 0 and lv.n_error == 0 for lv in out.levels)
    assert verdicts(out.verdicts) == {0.5: 8, 0.3: 8}  # 0.01 s on 0.2 s clips: RTF ~0.05
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert (saved["mode"], saved["pause_s"]) == ("open", 0.05)
    assert all(lv.client_lag_max_s is not None and lv.client_lag_max_s < 0.05 for lv in out.levels)


async def test_open_run_fails_a_level_when_requests_fail(tmp_path: Path) -> None:
    # 0.05 s per request, a send every 0.2 s per speaker: one speaker never overlaps itself, but
    # 8 speakers' phases in a 0.2 s cycle always put two within 0.025 s, so requests overlap and
    # the second in flight fails
    fake = FakeTransport(latency=lambda n: 0.05, text=lambda wav, n: "a b", fail_at=2)
    config = cfg(tmp_path, [1, 8, 16, 32], mode="open", bisect=False)
    out = await run_loadtest(fake, CLIPS, config, "t", tmp_path / "run", fake_meta(), None)
    assert [lv.level for lv in out.levels] == [1, 8, 16]  # one level past the first failure
    assert out.levels[1].n_error > 0
    assert verdicts(out.verdicts) == {0.5: 1, 0.3: 1}


async def test_closed_run_summary_says_closed(tmp_path: Path) -> None:
    fake = FakeTransport(latency=lambda n: 0.001, text=lambda wav, n: "a b")
    config = cfg(tmp_path, [1])
    out = await run_loadtest(fake, CLIPS, config, "t", tmp_path / "r", fake_meta(), None)
    assert (out.mode, out.pause_s) == ("closed", 0.0)


class BusyServer(FakeTransport):
    """A fake vLLM that reports requests still running until ``idle_at`` (monotonic time)."""

    def __init__(self, idle_at: float) -> None:
        super().__init__(latency=lambda n: 0.001, text=lambda wav, n: "a b")
        self.idle_at = idle_at

    async def scrape_metrics(self) -> str | None:
        running = 2 if time.monotonic() < self.idle_at else 0
        return f"vllm:num_requests_running {running}\nvllm:num_requests_waiting 0\n"


def fast_metrics(config: LoadTestConfig) -> LoadTestConfig:
    return config.model_copy(update={"metrics_scrape_hz": 20.0})


async def test_each_level_waits_for_the_server_to_drain(tmp_path: Path) -> None:
    fake = BusyServer(idle_at=time.monotonic() + 0.3)
    config = fast_metrics(cfg(tmp_path, [1]))
    await run_loadtest(fake, CLIPS, config, "t", tmp_path / "r", fake_meta(), None)
    rows = [
        json.loads(line) for line in (tmp_path / "r" / "requests.jsonl").read_text().splitlines()
    ]
    assert min(row["sent"] for row in rows) >= fake.idle_at
    level = json.loads((tmp_path / "r" / "levels.jsonl").read_text().splitlines()[0])
    assert 0.25 <= level["drain_s"] < 1.0


async def test_drain_gives_up_after_the_request_timeout(tmp_path: Path) -> None:
    fake = BusyServer(idle_at=float("inf"))  # never idle: the level still runs after 2 s
    config = fast_metrics(cfg(tmp_path, [1]))
    out = await run_loadtest(fake, CLIPS, config, "t", tmp_path / "r", fake_meta(), None)
    assert out.levels[0].n_ok > 0
    assert 2.0 <= (out.levels[0].drain_s or 0.0) < 3.0


async def test_servers_without_metrics_are_not_drained(tmp_path: Path) -> None:
    fake = FakeTransport(latency=lambda n: 0.001, text=lambda wav, n: "a b")
    out = await run_loadtest(
        fake, CLIPS, cfg(tmp_path, [1]), "t", tmp_path / "r", fake_meta(), None
    )
    assert out.levels[0].drain_s is None


def test_build_pool_encodes_wavs_with_references(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, [("a", 2.5, "2-5", "hello there")])
    (clip,) = build_pool(entries, 16000)
    assert (clip.id, clip.audio_s, clip.reference) == (entries[0].audio, 2.5, "hello there")
    assert clip.wav.startswith(b"RIFF")
