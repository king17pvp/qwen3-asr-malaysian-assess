"""Backend-neutral transcription contract shared by evaluation, benchmarking and serving."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from asr_assess.core.audio import Samples


@dataclass(frozen=True)
class AudioRequest:
    """One utterance to transcribe: 16 kHz mono samples and an optional language to force."""

    id: str
    samples: Samples
    language: str | None = None


@dataclass(frozen=True)
class Transcript:
    """An engine's output for one request, without the Qwen ``language X<asr_text>`` prefix."""

    id: str
    text: str
    language: str | None
    hit_token_limit: bool = False


class ASREngine(Protocol):
    """Turns a batch of requests into one transcript per request, in the same order."""

    name: str

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        """Transcribe every request in ``batch``."""
        ...


@runtime_checkable
class ReportsPeakMemory(Protocol):
    """Engines that can report their peak accelerator memory."""

    def peak_memory_gb(self) -> float:
        """Peak memory allocated since the last reset, in GiB."""
        ...

    def reset_peak_memory(self) -> None:
        """Start a new peak-memory measurement window."""
        ...


def check_alignment(batch: Sequence[AudioRequest], transcripts: Sequence[Transcript]) -> None:
    """Raise unless ``transcripts`` match ``batch`` one-to-one, in order."""
    want, got = [r.id for r in batch], [t.id for t in transcripts]
    if want != got:
        raise RuntimeError(f"Engine returned ids {got[:3]}... for batch ids {want[:3]}...")
