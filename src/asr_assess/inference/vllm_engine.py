"""vLLM backend over its OpenAI-compatible transcription endpoint (`vllm serve`).

The endpoint returns only the transcript: the detected language and the finish reason are not
exposed, so ``language`` is None and ``hit_token_limit`` is always False.
"""

from collections.abc import Sequence
from typing import Any

from asr_assess.core.config import VLLMHTTPEngineConfig
from asr_assess.core.transcription_api import (
    ENDPOINT,
    FILE_FIELD,
    encode_wav,
    form_fields,
    parse_transcription,
)
from asr_assess.inference.engine import AudioRequest, Transcript


class VLLMHTTPEngine:
    """Sends each request to ``vllm serve``, one at a time, in order."""

    def __init__(
        self, cfg: VLLMHTTPEngineConfig, sample_rate: int = 16000, client: Any | None = None
    ) -> None:
        import httpx

        self.name = f"vllm_http:{cfg.model or 'default'}@{cfg.base_url}"
        self._sample_rate = sample_rate
        self._client: Any = client or httpx.Client(
            base_url=cfg.base_url, timeout=cfg.request_timeout_s
        )
        self._fields = form_fields(cfg.model, cfg.max_new_tokens)

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        """One POST per request; raises on HTTP errors or unusable bodies."""
        return [Transcript(r.id, self._post(r), None) for r in batch]

    def _post(self, request: AudioRequest) -> str:
        wav = encode_wav(request.samples, self._sample_rate)
        files = {FILE_FIELD: (f"{request.id}.wav", wav, "audio/wav")}
        response = self._client.post(ENDPOINT, data=self._fields, files=files)
        response.raise_for_status()
        return parse_transcription(response.json())
