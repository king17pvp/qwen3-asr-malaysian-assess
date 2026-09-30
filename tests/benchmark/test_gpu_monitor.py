"""Tests for the background GPU sampler."""

import time

from asr_assess.benchmark.gpu_monitor import GpuMonitor, GpuSample, summarize_gpu


class StepReader:
    def __init__(self) -> None:
        self.n = 0

    def read(self) -> tuple[float, float]:
        self.n += 1
        return float(self.n), 2.0


class BrokenReader:
    def read(self) -> tuple[float, float]:
        raise RuntimeError("NVML went away")


def test_summary_uses_only_the_window() -> None:
    samples = [GpuSample(t, u, m) for t, u, m in [(0, 100, 9), (1, 20, 3), (2, 40, 5), (3, 100, 9)]]
    window = summarize_gpu(samples, 1.0, 3.0)
    assert window is not None
    assert (window.mean_util_pct, window.peak_util_pct) == (30.0, 40.0)
    assert (window.mean_mem_gib, window.peak_mem_gib) == (4.0, 5.0)


def test_empty_window_is_none() -> None:
    assert summarize_gpu([GpuSample(0, 1, 1)], 5.0, 6.0) is None


def test_monitor_samples_in_background() -> None:
    with GpuMonitor(StepReader(), hz=200) as monitor:
        time.sleep(0.1)
    assert len(monitor.samples()) >= 5


def test_a_failing_reader_stops_sampling_without_raising() -> None:
    with GpuMonitor(BrokenReader(), hz=200) as monitor:
        time.sleep(0.05)
    assert monitor.samples() == []
