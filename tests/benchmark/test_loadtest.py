"""End-to-end load-test driver tests against a fake server whose latency grows with load."""

import json
from pathlib import Path

import numpy as np
import pytest

from asr_assess.benchmark.load_stats import PoolClip
from asr_assess.benchmark.loadtest import build_pool, run_loadtest
from asr_assess.core.config import LoadTestConfig
from asr_assess.core.transcription_api import encode_wav
from tests.fakes import FakeTransport, fake_meta, write_clips

AUDIO_S = 0.2
WAV = encode_wav(np.zeros(3200, dtype=np.float32), 16000)
CLIPS = [PoolClip(f"c{i}", AUDIO_S, "2-5", "a b", WAV) for i in range(5)]


def cfg(tmp: Path, levels: list[int]) -> LoadTestConfig:
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


def test_build_pool_encodes_wavs_with_references(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, [("a", 2.5, "2-5", "hello there")])
    (clip,) = build_pool(entries, 16000)
    assert (clip.id, clip.audio_s, clip.reference) == (entries[0].audio, 2.5, "hello there")
    assert clip.wav.startswith(b"RIFF")
