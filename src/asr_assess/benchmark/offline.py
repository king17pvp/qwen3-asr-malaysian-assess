"""Single-stream RTF per duration bucket: one request at a time, batch 1, after warm-up."""

import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from asr_assess.benchmark.env_info import EnvInfo
from asr_assess.core.config import BenchConfig, LanguageHint
from asr_assess.core.manifest import ManifestEntry
from asr_assess.core.metrics import RTFStats, rtf, summarize_rtf
from asr_assess.core.run_record import RunRecord
from asr_assess.inference.engine import ASREngine, AudioRequest, ReportsPeakMemory, check_alignment
from asr_assess.inference.requests import to_requests

log = logging.getLogger(__name__)

Clock = Callable[[], float]


@dataclass(frozen=True)
class Timing:
    """Processing time of one measured request."""

    id: str
    bucket: str
    audio_s: float
    processing_s: float
    repeat: int

    @property
    def rtf(self) -> float:
        """Real-time factor of this request."""
        return rtf(self.processing_s, self.audio_s)


@dataclass(frozen=True)
class BenchSummary:
    """RTF distribution overall and per bucket, single-stream throughput and peak memory."""

    overall: RTFStats
    by_bucket: dict[str, RTFStats]
    audio_s_per_s: float
    peak_memory_gb: float | None


def select_bench_set(
    entries: Sequence[ManifestEntry], clips_per_bucket: int, seed: int
) -> list[ManifestEntry]:
    """Up to ``clips_per_bucket`` random clips from each bucket, seeded."""
    by_bucket: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        by_bucket[entry.bucket].append(entry)
    rng = random.Random(seed)
    chosen: list[ManifestEntry] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket], key=lambda e: e.audio)
        chosen.extend(rng.sample(pool, min(clips_per_bucket, len(pool))))
    return chosen


def measure(
    engine: ASREngine,
    requests: Sequence[AudioRequest],
    entries: Sequence[ManifestEntry],
    warmup: int,
    repeats: int,
    clock: Clock,
) -> list[Timing]:
    """Time each request alone, ``repeats`` times, after ``warmup`` discarded calls."""
    for i in range(warmup):
        engine.transcribe([requests[i % len(requests)]])
    timings = []
    for repeat in range(repeats):
        for request, entry in zip(requests, entries, strict=True):
            start = clock()
            check_alignment([request], engine.transcribe([request]))
            elapsed = clock() - start
            timings.append(Timing(entry.audio, entry.bucket, entry.duration, elapsed, repeat))
    return timings


def summarize(timings: Sequence[Timing], peak_memory_gb: float | None) -> BenchSummary:
    """Per-bucket and overall RTF statistics and audio seconds processed per second."""
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for timing in timings:
        by_bucket[timing.bucket].append(timing.rtf)
    busy_s = sum(t.processing_s for t in timings)
    return BenchSummary(
        overall=summarize_rtf([t.rtf for t in timings]),
        by_bucket={b: summarize_rtf(v) for b, v in sorted(by_bucket.items())},
        audio_s_per_s=sum(t.audio_s for t in timings) / busy_s if busy_s > 0 else 0.0,
        peak_memory_gb=peak_memory_gb,
    )


def run_offline(
    engine: ASREngine,
    entries: Sequence[ManifestEntry],
    cfg: BenchConfig,
    language_hint: LanguageHint,
    out_dir: Path,
    record: RunRecord,
    env: EnvInfo,
    clock: Clock = time.perf_counter,
) -> BenchSummary:
    """Benchmark a seeded bench set and write timings.jsonl and summary.json."""
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    chosen = select_bench_set(entries, cfg.clips_per_bucket, cfg.seed)
    requests = to_requests(chosen, cfg.sample_rate, language_hint)
    memory = engine if isinstance(engine, ReportsPeakMemory) else None
    if memory is not None:
        memory.reset_peak_memory()
    timings = measure(engine, requests, chosen, cfg.warmup_requests, cfg.repeats, clock)
    summary = summarize(timings, memory.peak_memory_gb() if memory is not None else None)
    out_dir.mkdir(parents=True)  # only once there is something to write
    with (out_dir / "timings.jsonl").open("w", encoding="utf-8") as handle:
        for timing in timings:
            handle.write(json.dumps(asdict(timing) | {"rtf": timing.rtf}) + "\n")
    payload = {"record": record.model_dump(mode="json"), "engine": engine.name,
               "env": asdict(env), "summary": asdict(summary)}  # fmt: skip
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("P95 RTF %.3f over %d requests -> %s", summary.overall.p95, len(timings), out_dir)
    return summary
