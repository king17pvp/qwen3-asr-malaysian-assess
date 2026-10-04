"""Schema of load-test result files: written by the benchmark, read by reporting."""

from pathlib import Path
from typing import Any, Literal

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


class RTFSummary(StrictModel):
    """Distribution of per-request real-time factors (mirrors core.metrics.RTFStats)."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


class LevelResult(StrictModel):
    """Everything measured at one concurrency level (pooled over its repeats)."""

    level: int
    repeats: int
    n_sent: int
    n_ok: int
    n_timeout: int
    n_error: int
    rtf: RTFSummary | None  # None when no request succeeded
    requests_per_s: float
    audio_s_per_s: float
    wer: float | None
    cer: float | None
    gpu: GpuWindow | None
    vllm: dict[str, GaugeStats] | None
    # open loop: how late the client sent requests after their scheduled time (event-loop lag,
    # which RTF does not include); None for closed loop
    client_lag_p95_s: float | None = None
    client_lag_max_s: float | None = None


class Verdict(StrictModel):
    """Highest sustainable concurrency at one P95 RTF threshold (None: not even 1 stream)."""

    threshold: float
    max_sustainable: int | None


class LoadRunSummary(StrictModel):
    """summary.json of one load-test run (one journey row)."""

    label: str
    profile: str
    mode: Literal["closed", "open"] = "closed"  # client model of the profile (LoadProfile.mode)
    pause_s: float = 0.0  # open loop: pause between a speaker's utterances
    url: str
    server_config: str | None  # the server YAML's text, as run
    record: dict[str, Any]
    env: dict[str, Any]
    reference_wer: float | None  # level-1 WER of this run
    levels: list[LevelResult]
    verdicts: list[Verdict]


def read_summary(path: Path) -> LoadRunSummary:
    """Load and validate a summary.json."""
    return LoadRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
