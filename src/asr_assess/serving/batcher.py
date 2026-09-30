"""Dynamic batching in front of an ASREngine: up to max_batch requests or max_wait, one GPU worker.

With ``max_batch=1`` and ``max_wait_s=0`` this is the baseline: one request at a time, FIFO.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from asr_assess.inference.engine import ASREngine, AudioRequest, Transcript, check_alignment

log = logging.getLogger(__name__)
_Pending = tuple[AudioRequest, "asyncio.Future[Transcript]"]


class Batcher:
    """Collects concurrent requests into batches; GPU work runs on one worker thread, in order."""

    def __init__(self, engine: ASREngine, max_batch: int, max_wait_s: float) -> None:
        self._engine, self._max_batch, self._max_wait_s = engine, max_batch, max_wait_s
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._task: asyncio.Task[None] | None = None
        self.batch_sizes: list[int] = []

    async def start(self) -> None:
        """Start the collector loop."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop collecting and release the worker thread."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._executor.shutdown(wait=True)

    async def submit(self, request: AudioRequest) -> Transcript:
        """Queue one request and wait for its transcript."""
        future: asyncio.Future[Transcript] = asyncio.get_running_loop().create_future()
        await self._queue.put((request, future))
        return await future

    async def _collect(self) -> list[_Pending]:
        """The first queued request plus whatever arrives within max_wait, up to max_batch."""
        batch = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait_s
        while len(batch) < self._max_batch:
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    item = self._queue.get_nowait()
                else:
                    item = await asyncio.wait_for(self._queue.get(), remaining)
            except (asyncio.QueueEmpty, TimeoutError):
                break
            batch.append(item)
        return [(r, f) for r, f in batch if not f.done()]  # drop callers that gave up

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            if not batch:
                continue
            requests = [r for r, _ in batch]
            self.batch_sizes.append(len(requests))
            try:
                out = await loop.run_in_executor(self._executor, self._engine.transcribe, requests)
                check_alignment(requests, out)
            except Exception as err:  # the engine's failure belongs to this batch only
                log.exception("Batch of %d failed", len(requests))
                for _, future in batch:
                    if not future.done():
                        future.set_exception(err)
                continue
            for (_, future), transcript in zip(batch, out, strict=True):
                if not future.done():
                    future.set_result(transcript)
