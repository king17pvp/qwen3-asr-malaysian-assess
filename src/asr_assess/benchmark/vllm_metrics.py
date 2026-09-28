"""vLLM scheduler gauges (running, waiting, KV-cache usage) from its Prometheus /metrics text."""

from collections import defaultdict
from collections.abc import Sequence

from asr_assess.core.load_results import GaugeStats

# Our name -> vLLM metric names, newest first (the KV gauge was renamed across versions).
GAUGES: dict[str, tuple[str, ...]] = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
_ALIAS = {metric: ours for ours, metrics in GAUGES.items() for metric in metrics}

Scrape = tuple[float, dict[str, float]]


def parse_gauges(text: str) -> dict[str, float]:
    """Our gauges from Prometheus text, summed over label sets (e.g. several engines)."""
    values: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        ours = _ALIAS.get(series.split("{", 1)[0])
        if ours is not None:
            values[ours] += float(raw)
    return dict(values)


def summarize_gauges(scrapes: Sequence[Scrape], t0: float, t1: float) -> dict[str, GaugeStats]:
    """Mean and max of each gauge over scrapes taken in ``[t0, t1)``."""
    series: dict[str, list[float]] = defaultdict(list)
    for t, gauges in scrapes:
        if t0 <= t < t1:
            for name, value in gauges.items():
                series[name].append(value)
    return {n: GaugeStats(mean=sum(v) / len(v), max=max(v)) for n, v in sorted(series.items())}
