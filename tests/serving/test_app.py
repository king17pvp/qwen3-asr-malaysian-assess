"""Tests for the HF server's HTTP layer with a fake engine."""

import io
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from asr_assess.core.transcription_api import encode_wav
from asr_assess.inference.engine import AudioRequest, Transcript
from asr_assess.serving.app import create_app
from asr_assess.serving.batcher import Batcher

URL = "/v1/audio/transcriptions"


class LengthEngine:
    """Echoes the decoded sample count, so tests can check decoding and resampling."""

    name = "length"

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        return [Transcript(r.id, str(len(r.samples)), None) for r in batch]


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Batcher(LengthEngine(), 1, 0.0), sample_rate=16000)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_transcribes_16k_wav_in_openai_shape(client: TestClient) -> None:
    wav = encode_wav(np.zeros(16000, dtype=np.float32), 16000)
    response = client.post(
        URL,
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "x", "temperature": "0", "response_format": "json"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "16000"}


def test_resamples_stereo_44k_upload(client: TestClient) -> None:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((44100, 2), dtype=np.float32), 44100, format="WAV")
    response = client.post(URL, files={"file": ("a.wav", buffer.getvalue(), "audio/wav")})
    assert response.json() == {"text": "16000"}


def test_garbage_upload_is_400(client: TestClient) -> None:
    response = client.post(URL, files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400


def test_missing_file_is_422(client: TestClient) -> None:
    assert client.post(URL, data={"model": "x"}).status_code == 422
