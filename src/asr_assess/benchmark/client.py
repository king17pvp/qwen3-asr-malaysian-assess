"""How the load test reaches a server: a Transport Protocol and the OpenAI transcription client.

Both our HF server and `vllm serve` expose the same endpoint, so one implementation serves every
journey row; tests use a fake.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from asr_assess.core.transcription_api import (
    ENDPOINT,
    FILE_FIELD,
    HEALTH,
    METRICS,
    form_fields,
    parse_transcription,
)

_HTTP_OK = 200


@dataclass(frozen=True)
class TransportResult:
    """Outcome of one request: status is "ok", "timeout", "error" or "http_<code>"."""

    status: str
    text: str = ""


class Transport(Protocol):
    """A server the load test can send utterances to."""

    async def transcribe(self, wav: bytes) -> TransportResult:
        """Send one WAV and wait for its transcript; never raises for server-side failures."""
        ...

    async def wait_ready(self, timeout_s: float) -> bool:
        """Whether the server became healthy within ``timeout_s``."""
        ...

    async def scrape_metrics(self) -> str | None:
        """Prometheus text from the server, or None when it has no metrics endpoint."""
        ...

    async def aclose(self) -> None:
        """Release connections."""
        ...


class OpenAITranscriptionTransport:
    """``POST /v1/audio/transcriptions`` over one pooled async HTTP client."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        max_tokens: int | None,
        model: str | None = None,
        client: Any | None = None,
        poll_s: float = 1.0,
    ) -> None:
        import httpx

        self._httpx: Any = httpx
        # No connection cap: every stream must get its own connection, or the client queues.
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        self._client: Any = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s, limits=limits
        )
        self._fields = form_fields(model, max_tokens)
        self._poll_s = poll_s

    async def transcribe(self, wav: bytes) -> TransportResult:
        """Send one WAV; timeouts, HTTP errors and unusable bodies become statuses."""
        files = {FILE_FIELD: ("clip.wav", wav, "audio/wav")}
        try:
            response = await self._client.post(ENDPOINT, data=self._fields, files=files)
        except self._httpx.TimeoutException:
            return TransportResult("timeout")
        except self._httpx.HTTPError:
            return TransportResult("error")
        if not response.is_success:
            return TransportResult(f"http_{response.status_code}")
        try:
            return TransportResult("ok", parse_transcription(response.json()))
        except (ValueError, json.JSONDecodeError):
            return TransportResult("error")

    async def wait_ready(self, timeout_s: float) -> bool:
        """Poll the health endpoint until it answers 200 or ``timeout_s`` passes."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if (await self._client.get(HEALTH)).status_code == _HTTP_OK:
                    return True
            except self._httpx.HTTPError:
                pass
            await asyncio.sleep(self._poll_s)
        return False

    async def scrape_metrics(self) -> str | None:
        """The metrics text, or None on any non-200 answer or connection failure."""
        try:
            response = await self._client.get(METRICS)
        except self._httpx.HTTPError:
            return None
        return str(response.text) if response.status_code == _HTTP_OK else None

    async def aclose(self) -> None:
        """Close the pooled client."""
        await self._client.aclose()
