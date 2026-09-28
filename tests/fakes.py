"""Test doubles shared across test packages: engines, a clock, a server transport, WAV fixtures."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from asr_assess.benchmark.client import TransportResult
from asr_assess.core.audio import write_wav
from asr_assess.core.manifest import ManifestEntry
from asr_assess.inference.engine import AudioRequest, Transcript

SAMPLE_RATE = 16000


class FakeClock:
    """A monotonic clock that only moves when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedEngine:
    """Returns scripted text per request id; advances ``clock`` by ``rtf`` x audio seconds."""

    def __init__(
        self,
        texts: Mapping[str, str] | None = None,
        clock: FakeClock | None = None,
        rtf: float = 0.1,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._texts = dict(texts or {})
        self._clock = clock
        self._rtf = rtf
        self.calls: list[list[AudioRequest]] = []

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        self.calls.append(list(batch))
        if self._clock is not None:
            audio_s = sum(len(r.samples) for r in batch) / SAMPLE_RATE
            self._clock.advance(self._rtf * audio_s)
        return [Transcript(r.id, self._texts.get(r.id, ""), r.language) for r in batch]


class FakeTransport:
    """A server whose latency depends on how many requests are in flight (itself included)."""

    def __init__(
        self,
        latency: Callable[[int], float],
        text: Callable[[bytes, int], str] | None = None,
        fail_at: int | None = None,
    ) -> None:
        self._latency, self._text, self._fail_at = latency, text, fail_at
        self.inflight = 0
        self.max_inflight = 0

    async def transcribe(self, wav: bytes) -> TransportResult:
        self.inflight += 1
        seen = self.inflight
        self.max_inflight = max(self.max_inflight, seen)
        try:
            await asyncio.sleep(self._latency(seen))
        finally:
            self.inflight -= 1
        if self._fail_at is not None and seen >= self._fail_at:
            return TransportResult("http_500")
        return TransportResult("ok", self._text(wav, seen) if self._text else "")

    async def wait_ready(self, timeout_s: float) -> bool:
        return True

    async def scrape_metrics(self) -> str | None:
        return None

    async def aclose(self) -> None:
        return None


def write_clips(
    tmp_path: Path, specs: Sequence[tuple[str, float, str, str]]
) -> list[ManifestEntry]:
    """Write silent WAVs; each spec is (name, seconds, bucket, transcript)."""
    entries = []
    for name, seconds, bucket, text in specs:
        path = tmp_path / f"{name}.wav"
        write_wav(path, np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
        entries.append(
            ManifestEntry(
                audio=str(path),
                text=f"language Malay<asr_text>{text}",
                duration=seconds,
                bucket=bucket,
                source=f"cat_{bucket}/src",
            )
        )
    return entries
