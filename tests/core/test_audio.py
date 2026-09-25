"""Tests for audio loading/resampling and duration bucketing."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_assess.core.audio import (
    Bucket,
    assign_bucket,
    decode_audio,
    duration_s,
    load_audio,
    write_wav,
)

BUCKETS = (Bucket("2-5", 2.0, 5.0), Bucket("5-15", 5.0, 15.0), Bucket("15-30", 15.0, 30.0))


class TestAssignBucket:
    @pytest.mark.parametrize(
        ("duration", "expected"),
        [(2.0, "2-5"), (4.99, "2-5"), (5.0, "5-15"), (14.9, "5-15"), (15.0, "15-30")],
    )
    def test_lower_edge_is_inclusive(self, duration: float, expected: str) -> None:
        assert assign_bucket(duration, BUCKETS) == expected

    def test_last_bucket_includes_upper_edge(self) -> None:
        assert assign_bucket(30.0, BUCKETS) == "15-30"

    @pytest.mark.parametrize("duration", [0.5, 1.99, 30.01])
    def test_out_of_range_is_none(self, duration: float) -> None:
        assert assign_bucket(duration, BUCKETS) is None

    def test_bucket_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError):
            Bucket("bad", 5.0, 2.0)


class TestAudioIO:
    def test_load_resamples_and_downmixes(self, tmp_path: Path) -> None:
        path = tmp_path / "stereo_8k.wav"
        sr_in = 8000
        stereo = np.zeros((sr_in * 2, 2), dtype=np.float32)
        sf.write(path, stereo, sr_in)

        samples = load_audio(path, sample_rate=16000)

        assert samples.ndim == 1
        assert samples.dtype == np.float32
        assert duration_s(samples, 16000) == pytest.approx(2.0, abs=1e-3)

    def test_write_then_load_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "tone.wav"
        tone = (0.1 * np.sin(np.linspace(0, 200 * np.pi, 16000))).astype(np.float32)

        write_wav(path, tone, sample_rate=16000)

        assert np.allclose(load_audio(path, sample_rate=16000), tone, atol=1e-3)

    def test_decode_bytes_resamples_and_downmixes(self) -> None:
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros((22050, 2), dtype=np.float32), 22050, format="WAV")

        samples = decode_audio(buffer.getvalue(), sample_rate=16000)

        assert samples.ndim == 1
        assert duration_s(samples, 16000) == pytest.approx(1.0, abs=1e-3)

    def test_duration_rejects_nonpositive_rate(self) -> None:
        with pytest.raises(ValueError):
            duration_s(np.zeros(10, dtype=np.float32), 0)
