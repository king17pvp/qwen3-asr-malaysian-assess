"""The one HTTP transcription contract (OpenAI style) used by vLLM, our HF server and clients."""

import io

import soundfile as sf

from asr_assess.core.audio import Samples

ENDPOINT = "/v1/audio/transcriptions"
HEALTH = "/health"
METRICS = "/metrics"
FILE_FIELD = "file"
_TAG_BITS = 64


def encode_wav(samples: Samples, sample_rate: int) -> bytes:
    """Mono samples as an in-memory 16-bit PCM WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def tag_wav(wav: bytes, tag: int) -> bytes:
    """A copy of a 16-bit PCM WAV whose first 64 samples carry ``tag`` in their lowest bit.

    Gives every request unique audio (each sample moves by at most 1 LSB), so a server cannot reuse
    caches keyed by audio content, such as vLLM's multimodal and prefix caches, across repeats of a
    clip. Real traffic never repeats audio.
    """
    start = wav.find(b"data", 12) + 8  # skip the chunk id and its size
    if not wav.startswith(b"RIFF") or start < 8 or len(wav) - start < 2 * _TAG_BITS:
        raise ValueError("not a 16-bit PCM WAV with at least 64 samples")
    out = bytearray(wav)
    for bit in range(_TAG_BITS):
        low = start + 2 * bit  # little-endian: the low byte holds the lowest bit
        out[low] = (out[low] & 0xFE) | ((tag >> bit) & 1)
    return bytes(out)


def form_fields(model: str | None, max_tokens: int | None) -> dict[str, str]:
    """Multipart form fields besides the file: greedy decoding, JSON response."""
    fields = {"response_format": "json", "temperature": "0"}
    if model is not None:
        fields["model"] = model
    if max_tokens is not None:
        fields["max_completion_tokens"] = str(max_tokens)
    return fields


def parse_transcription(payload: object) -> str:
    """The transcript from a JSON response body; ValueError if it has no string ``text``."""
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise ValueError(f"Response has no string 'text': {str(payload)[:200]}")
    return text.strip()
