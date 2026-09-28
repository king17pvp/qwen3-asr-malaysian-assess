"""Tests for the one transcription HTTP contract shared by clients and the HF server."""

import numpy as np
import pytest

from asr_assess.core.audio import decode_audio
from asr_assess.core.transcription_api import encode_wav, form_fields, parse_transcription


def test_wav_round_trip_keeps_length_and_rate() -> None:
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    back = decode_audio(encode_wav(samples, 16000), 16000)
    assert len(back) == 16000
    assert np.allclose(back, samples, atol=1e-3)  # 16-bit PCM quantization


def test_form_fields_are_greedy_json_and_omit_unset() -> None:
    assert form_fields(None, None) == {"response_format": "json", "temperature": "0"}
    assert form_fields("m", 256) == {
        "response_format": "json",
        "temperature": "0",
        "model": "m",
        "max_completion_tokens": "256",
    }


def test_parse_transcription_reads_text() -> None:
    assert parse_transcription({"text": " hello ", "usage": {}}) == "hello"


@pytest.mark.parametrize("payload", [None, [], {"txt": "x"}, {"text": 3}])
def test_parse_transcription_rejects_bad_payloads(payload: object) -> None:
    with pytest.raises(ValueError, match="text"):
        parse_transcription(payload)
