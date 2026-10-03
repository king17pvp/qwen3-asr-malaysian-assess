"""Tests for the one transcription HTTP contract shared by clients and the HF server."""

import numpy as np
import pytest

from asr_assess.core.audio import decode_audio
from asr_assess.core.transcription_api import (
    encode_wav,
    form_fields,
    parse_transcription,
    tag_wav,
)


def test_wav_round_trip_keeps_length_and_rate() -> None:
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    back = decode_audio(encode_wav(samples, 16000), 16000)
    assert len(back) == 16000
    assert np.allclose(back, samples, atol=1e-3)  # 16-bit PCM quantization


def test_tag_wav_makes_audio_unique_within_one_lsb() -> None:
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    wav = encode_wav(samples, 16000)
    a, b = tag_wav(wav, 1), tag_wav(wav, 2**63)
    assert len({wav, a, b}) == 3
    assert len(a) == len(wav)
    pcm = [np.frombuffer(w[44:], dtype="<i2").astype(int) for w in (wav, a, b)]
    assert all(np.abs(p - pcm[0]).max() <= 1 for p in pcm[1:])
    assert np.allclose(decode_audio(a, 16000), samples, atol=1e-3)


def test_tag_wav_rejects_non_wav() -> None:
    with pytest.raises(ValueError, match="WAV"):
        tag_wav(b"RIFF", 1)


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
