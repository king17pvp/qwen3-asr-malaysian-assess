"""FastAPI app for the HF engine: the OpenAI-style transcription endpoint over the batcher.

Form fields other than ``file`` (model, language, temperature, response_format,
max_completion_tokens) are accepted and ignored: decoding settings come from the engine config.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from asr_assess.core.audio import decode_audio
from asr_assess.core.transcription_api import ENDPOINT, HEALTH
from asr_assess.inference.engine import AudioRequest
from asr_assess.serving.batcher import Batcher

_BAD_REQUEST = 400


def create_app(batcher: Batcher, sample_rate: int) -> FastAPI:
    """An app whose lifespan starts and stops ``batcher``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="asr-assess HF server", lifespan=lifespan)

    @app.get(HEALTH)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(ENDPOINT)
    async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
        data = await file.read()
        try:
            # decoding and resampling are CPU work: keep them off the event loop
            samples = await asyncio.to_thread(decode_audio, data, sample_rate)
        except Exception as err:
            raise HTTPException(_BAD_REQUEST, "could not decode audio") from err
        transcript = await batcher.submit(AudioRequest(uuid4().hex, samples))
        return {"text": transcript.text}

    return app
