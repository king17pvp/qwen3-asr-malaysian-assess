"""GPU utilization and memory sampled by NVML in a background thread during a load test."""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from asr_assess.core.load_results import GpuWindow

log = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class GpuSample:
    """One reading: time, utilization in percent, memory used in GiB."""

    t: float
    util_pct: float
    mem_used_gib: float


class GpuReader(Protocol):
    """Reads (utilization %, used GiB) of one GPU."""

    def read(self) -> tuple[float, float]:
        """One reading."""
        ...


class PynvmlReader:
    """NVML reader for one device (needs the `http` extra and an NVIDIA driver)."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        self._nvml: Any = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read(self) -> tuple[float, float]:
        """Current utilization and memory used."""
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
        return float(util), float(used) / _BYTES_PER_GIB


class GpuMonitor:
    """Samples ``reader`` at ``hz`` while the context is open; stops quietly if reads fail."""

    def __init__(
        self, reader: GpuReader, hz: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._reader, self._period, self._clock = reader, 1.0 / hz, clock
        self._samples: list[GpuSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def samples(self) -> list[GpuSample]:
        """A copy of the samples so far."""
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            try:
                util, mem = self._reader.read()
            except Exception:  # a lost GPU reading must never kill the load test
                log.warning("GPU sampling stopped", exc_info=True)
                return
            with self._lock:
                self._samples.append(GpuSample(self._clock(), util, mem))


def summarize_gpu(samples: Sequence[GpuSample], t0: float, t1: float) -> GpuWindow | None:
    """Mean and peak utilization and memory over samples in ``[t0, t1)``; None if there are none."""
    window = [s for s in samples if t0 <= s.t < t1]
    if not window:
        return None
    utils, mems = [s.util_pct for s in window], [s.mem_used_gib for s in window]
    return GpuWindow(
        mean_util_pct=sum(utils) / len(utils),
        peak_util_pct=max(utils),
        mean_mem_gib=sum(mems) / len(mems),
        peak_mem_gib=max(mems),
    )
