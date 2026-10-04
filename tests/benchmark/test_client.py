"""Tests for the load-test HTTP transport."""

import asyncio

import httpx
import pytest

from asr_assess.benchmark.client import (
    LeastOutstandingTransport,
    OpenAITranscriptionTransport,
    TransportResult,
)


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


async def test_connection_error_names_the_exception() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("disconnected", request=request)

    status = (await transport(httpx.MockTransport(handle)).transcribe(b"x")).status
    assert status == "error:RemoteProtocolError"


@pytest.mark.parametrize(
    "response", [httpx.Response(200, text="<html>"), httpx.Response(200, json={"no": "text"})]
)
async def test_bad_body_is_an_error_not_a_crash(response: httpx.Response) -> None:
    t = transport(httpx.MockTransport(lambda r: response))
    assert (await t.transcribe(b"x")).status == "error:bad_body"


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


class FakeServer:
    """A Transport that holds each request until released, counting what it saw."""

    def __init__(self, ready: bool = True, metrics: str | None = None) -> None:
        self.sent = 0
        self.closed = False
        self.ready, self.metrics = ready, metrics
        self.release = asyncio.Event()

    async def transcribe(self, wav: bytes) -> TransportResult:
        self.sent += 1
        await self.release.wait()
        return TransportResult("ok", "t")

    async def wait_ready(self, timeout_s: float) -> bool:
        return self.ready

    async def scrape_metrics(self) -> str | None:
        return self.metrics

    async def aclose(self) -> None:
        self.closed = True


async def test_least_outstanding_spreads_concurrent_requests() -> None:
    a, b = FakeServer(), FakeServer()
    spread = LeastOutstandingTransport([a, b])
    tasks = [asyncio.create_task(spread.transcribe(b"x")) for _ in range(4)]
    await asyncio.sleep(0)
    assert (a.sent, b.sent) == (2, 2)
    a.release.set()
    b.release.set()
    assert all(r.status == "ok" for r in await asyncio.gather(*tasks))


async def test_least_outstanding_prefers_the_idle_server() -> None:
    a, b = FakeServer(), FakeServer()
    spread = LeastOutstandingTransport([a, b])
    busy = asyncio.create_task(spread.transcribe(b"x"))
    await asyncio.sleep(0)
    b.release.set()
    await spread.transcribe(b"x")
    await spread.transcribe(b"x")
    assert (a.sent, b.sent) == (1, 2)
    a.release.set()
    await busy


async def test_least_outstanding_is_ready_only_when_every_server_is() -> None:
    assert await LeastOutstandingTransport([FakeServer(), FakeServer()]).wait_ready(1.0)
    assert not await LeastOutstandingTransport([FakeServer(), FakeServer(ready=False)]).wait_ready(
        1.0
    )


async def test_least_outstanding_joins_metrics_and_closes_all() -> None:
    a, b, c = FakeServer(metrics="m 1"), FakeServer(metrics=None), FakeServer(metrics="m 2")
    spread = LeastOutstandingTransport([a, b, c])
    assert await spread.scrape_metrics() == "m 1\nm 2"
    assert await LeastOutstandingTransport([FakeServer()]).scrape_metrics() is None
    await spread.aclose()
    assert a.closed and b.closed and c.closed


def test_least_outstanding_needs_a_server() -> None:
    with pytest.raises(ValueError):
        LeastOutstandingTransport([])
