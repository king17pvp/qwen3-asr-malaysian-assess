"""Load-test clients in worker processes, so one Python event loop does not cap the offered load.

One process playing hundreds of open-loop speakers falls behind its own timetable; past that
point the load generator, not the server, sets the limit. A ``WorkerPool`` runs each level's
clients split over several processes, each with its own transport; the parent keeps the sweep,
the drain, GPU and metrics sampling, and the results.
"""

import asyncio
import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from types import TracebackType
from typing import Self

from asr_assess.benchmark.client import Transport
from asr_assess.benchmark.load_stats import PoolClip, RequestRecord
from asr_assess.benchmark.loadtest import ClientPlan, run_plans

TransportFactory = Callable[[], Transport]

# Per worker process, set once by the pool's initializer.
_clips: Sequence[PoolClip] = ()
_factory: TransportFactory | None = None


def _init(clips: Sequence[PoolClip], factory: TransportFactory) -> None:
    global _clips, _factory
    _clips, _factory = clips, factory


def _ping(hold_s: float) -> int:
    time.sleep(hold_s)  # long enough that every worker takes one ping
    return os.getpid()


def _run(plans: Sequence[ClientPlan]) -> list[RequestRecord]:
    return asyncio.run(_run_async(plans))


async def _run_async(plans: Sequence[ClientPlan]) -> list[RequestRecord]:
    if _factory is None:
        raise RuntimeError("worker used before its initializer ran")
    transport = _factory()
    try:
        return await run_plans(transport, _clips, plans, time.monotonic)
    finally:
        await transport.aclose()


class WorkerPool:
    """``size`` worker processes; each level's clients are dealt round-robin over them.

    ``factory`` builds each worker's transport and must be picklable (a class, or a
    ``functools.partial`` of a module-level function). The clips are sent once per worker.
    """

    def __init__(self, size: int, clips: Sequence[PoolClip], factory: TransportFactory) -> None:
        if size < 1:
            raise ValueError("a worker pool needs at least one worker")
        self.size = size
        self._pool = ProcessPoolExecutor(
            size,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=(list(clips), factory),
        )
        # start and initialize every worker now, so process start-up never eats into a timetable
        list(self._pool.map(_ping, [0.2] * size))

    async def run(self, plans: Sequence[ClientPlan]) -> list[RequestRecord]:
        """Every plan's records; plans hold absolute ``time.monotonic`` times."""
        loop = asyncio.get_running_loop()
        shares = [list(plans[w :: self.size]) for w in range(self.size)]
        jobs = [loop.run_in_executor(self._pool, _run, share) for share in shares if share]
        return [r for got in await asyncio.gather(*jobs) for r in got]

    def close(self) -> None:
        """Stop the worker processes."""
        self._pool.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
