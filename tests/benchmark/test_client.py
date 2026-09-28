"""Tests for the load-test HTTP transport."""

import httpx
import pytest

from asr_assess.benchmark.client import OpenAITranscriptionTransport


def transport(handler: httpx.MockTransport) -> OpenAITranscriptionTransport:
    client = httpx.AsyncClient(transport=handler, base_url="http://s")
    return OpenAITranscriptionTransport(
        "http://s", timeout_s=1, max_tokens=256, client=client, poll_s=0.0
    )


async def test_ok_returns_text() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert b"RIFF" in request.read()
        return httpx.Response(200, json={"text": "hi"})

    result = await transport(httpx.MockTransport(handle)).transcribe(b"RIFF")
    assert (result.status, result.text) == ("ok", "hi")


async def test_non_2xx_is_http_status() -> None:
    t = transport(httpx.MockTransport(lambda r: httpx.Response(503)))
    assert (await t.transcribe(b"RIFF")).status == "http_503"


async def test_timeout_is_timeout() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    assert (await transport(httpx.MockTransport(handle)).transcribe(b"x")).status == "timeout"


async def test_connection_error_is_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert (await transport(httpx.MockTransport(handle)).transcribe(b"x")).status == "error"


@pytest.mark.parametrize(
    "response", [httpx.Response(200, text="<html>"), httpx.Response(200, json={"no": "text"})]
)
async def test_bad_body_is_an_error_not_a_crash(response: httpx.Response) -> None:
    t = transport(httpx.MockTransport(lambda r: response))
    assert (await t.transcribe(b"x")).status == "error"


async def test_metrics_returns_text_or_none() -> None:
    ok = transport(httpx.MockTransport(lambda r: httpx.Response(200, text="vllm:x 1")))
    assert await ok.scrape_metrics() == "vllm:x 1"
    missing = transport(httpx.MockTransport(lambda r: httpx.Response(404)))
    assert await missing.scrape_metrics() is None


async def test_wait_ready_polls_health_until_200() -> None:
    codes = iter([503, 200])
    t = transport(httpx.MockTransport(lambda r: httpx.Response(next(codes))))
    assert await t.wait_ready(timeout_s=1.0)


async def test_wait_ready_gives_up() -> None:
    t = transport(httpx.MockTransport(lambda r: httpx.Response(503)))
    assert not await t.wait_ready(timeout_s=0.05)
