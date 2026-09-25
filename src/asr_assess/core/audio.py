"""Audio I/O at a fixed sample rate, and duration bucketing."""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

Samples = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Bucket:
    """A named duration range in seconds, lower edge inclusive."""

    name: str
    min_s: float
    max_s: float

    def __post_init__(self) -> None:
        if self.min_s >= self.max_s:
            raise ValueError(f"Bucket {self.name!r}: min_s must be below max_s")


def assign_bucket(duration: float, buckets: Sequence[Bucket]) -> str | None:
    """Name of the bucket holding ``duration``; the last bucket also includes its upper edge."""
    for i, bucket in enumerate(buckets):
        is_last = i == len(buckets) - 1
        if bucket.min_s <= duration < bucket.max_s or (is_last and duration == bucket.max_s):
            return bucket.name
    return None


def load_audio(path: Path, sample_rate: int) -> Samples:
    """Load an audio file as mono float32 at ``sample_rate``."""
    return _read_mono(path, sample_rate)


def decode_audio(data: bytes, sample_rate: int) -> Samples:
    """Decode in-memory audio (WAV, FLAC, MP3, ...) as mono float32 at ``sample_rate``."""
    return _read_mono(io.BytesIO(data), sample_rate)


def _read_mono(source: Path | io.BytesIO, sample_rate: int) -> Samples:
    data, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if source_rate != sample_rate:
        import librosa  # slow import; only paid when resampling is needed

        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    return np.ascontiguousarray(mono, dtype=np.float32)


def write_wav(path: Path, samples: Samples, sample_rate: int) -> None:
    """Write mono samples as a 16-bit PCM WAV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def duration_s(samples: Samples, sample_rate: int) -> float:
    """Duration in seconds of ``samples`` at ``sample_rate``."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    return len(samples) / sample_rate
