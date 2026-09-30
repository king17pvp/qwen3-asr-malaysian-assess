"""Tests for the dynamic batcher."""

import asyncio
import time
from collections.abc import Sequence

import numpy as np
import pytest

from asr_assess.inference.engine import AudioRequest, Transcript
from asr_assess.serving.batcher import Batcher
from tests.fakes import ScriptedEngine


def req(i: int) -> AudioRequest:
    return AudioRequest(f"r{i}", np.zeros(160, dtype=np.float32))


class SlowEngine(ScriptedEngine):
    """Blocks for ``delay_s`` per batch, like a GPU call; optionally fails."""

    def __init__(self, delay_s: float, fail: bool = False) -> None:
        super().__init__({f"r{i}": f"t{i}" for i in range(100)})
        self.delay_s, self.fail = delay_s, fail

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("gpu fell over")
        return super().transcribe(batch)


async def run(batcher: Batcher, n: int) -> list[str]:
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(req(i)) for i in range(n)))
    finally:
        await batcher.stop()
    return [t.text for t in results]


async def test_results_reach_their_callers() -> None:
    texts = await run(Batcher(SlowEngine(0.0), max_batch=4, max_wait_s=0.01), 10)
    assert texts == [f"t{i}" for i in range(10)]


async def test_batches_never_exceed_max_batch() -> None:
    batcher = Batcher(SlowEngine(0.02), max_batch=3, max_wait_s=0.05)
    await run(batcher, 10)
    assert max(batcher.batch_sizes) == 3
    assert sum(batcher.batch_sizes) == 10


async def test_max_batch_one_is_sequential() -> None:
    batcher = Batcher(SlowEngine(0.0), max_batch=1, max_wait_s=0.0)
    await run(batcher, 5)
    assert batcher.batch_sizes == [1] * 5


async def test_flushes_a_partial_batch_after_max_wait() -> None:
    batcher = Batcher(SlowEngine(0.0), max_batch=8, max_wait_s=0.05)
    await batcher.start()
    start = time.monotonic()
    await batcher.submit(req(0))
    elapsed = time.monotonic() - start
    await batcher.stop()
    assert 0.04 <= elapsed < 0.5
    assert batcher.batch_sizes == [1]


async def test_engine_error_fails_only_its_batch() -> None:
    engine = SlowEngine(0.0, fail=True)
    batcher = Batcher(engine, max_batch=2, max_wait_s=0.0)
    await batcher.start()
    with pytest.raises(RuntimeError, match="fell over"):
        await batcher.submit(req(0))
    engine.fail = False
    assert (await batcher.submit(req(1))).text == "t1"  # the collector is still alive
    await batcher.stop()


async def test_cancelled_caller_does_not_break_the_batcher() -> None:
    batcher = Batcher(SlowEngine(0.1), max_batch=4, max_wait_s=0.0)
    await batcher.start()
    task = asyncio.create_task(batcher.submit(req(0)))
    await asyncio.sleep(0.02)  # request 0 is now on the GPU thread
    task.cancel()
    assert (await batcher.submit(req(1))).text == "t1"
    await batcher.stop()
