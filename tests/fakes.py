"""Test doubles shared across test packages: a scripted ASR engine, a clock, WAV fixtures."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

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
