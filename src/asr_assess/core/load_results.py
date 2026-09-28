"""Schema of load-test result files: written by the benchmark, read by reporting."""

from asr_assess.core.config import StrictModel


class GpuWindow(StrictModel):
    """GPU utilization and memory over a measurement window."""

    mean_util_pct: float
    peak_util_pct: float
    mean_mem_gib: float
    peak_mem_gib: float


class GaugeStats(StrictModel):
    """Mean and max of a scraped server gauge over a measurement window."""

    mean: float
    max: float
