"""Tests for the vLLM HTTP engine against a mocked transport."""

import httpx
import numpy as np
import pytest

from asr_assess.core.config import VLLMHTTPEngineConfig
from asr_assess.inference.engine import AudioRequest
from asr_assess.inference.factory import make_engine
from asr_assess.inference.vllm_engine import VLLMHTTPEngine

CFG = VLLMHTTPEngineConfig(
    kind="vllm_http",
    base_url="http://vllm:8000",
    request_timeout_s=5,
    max_new_tokens=256,
    language_hint="auto",
)


def engine(handler: httpx.MockTransport) -> VLLMHTTPEngine:
    return VLLMHTTPEngine(CFG, client=httpx.Client(transport=handler, base_url=CFG.base_url))


def silence(i: int) -> AudioRequest:
    return AudioRequest(f"a{i}", np.zeros(1600, dtype=np.float32))


def test_posts_one_multipart_request_per_clip_in_order() -> None:
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        bodies.append(request.read())
        return httpx.Response(200, json={"text": f" t{len(bodies)} "})

    out = engine(httpx.MockTransport(handle)).transcribe([silence(0), silence(1)])
    assert [(t.id, t.text, t.language) for t in out] == [("a0", "t1", None), ("a1", "t2", None)]
    assert all(b'name="max_completion_tokens"' in b and b"RIFF" in b for b in bodies)


def test_server_error_raises() -> None:
    handler = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        engine(handler).transcribe([silence(0)])


def test_factory_builds_the_vllm_engine() -> None:
    assert make_engine(CFG).name == "vllm_http:default@http://vllm:8000"
