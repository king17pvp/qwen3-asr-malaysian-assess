"""Closed-loop N-stream load test: each stream sends one utterance, waits, then sends the next.

Per-request RTF = end-to-end latency (including queueing) / audio duration. A request counts when
it is *sent* inside the steady-state window, so slow requests finishing after it still count.
"""

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from asr_assess.benchmark.client import Transport
from asr_assess.benchmark.env_info import EnvInfo
from asr_assess.benchmark.gpu_monitor import GpuMonitor, summarize_gpu
from asr_assess.benchmark.load_stats import (
    PoolClip,
    RequestRecord,
    bisect_next,
    in_window,
    is_sustainable,
    max_sustainable,
    stream_order,
    summarize_level,
)
from asr_assess.benchmark.vllm_metrics import Scrape, parse_gauges, summarize_gauges
from asr_assess.core.audio import load_audio
from asr_assess.core.config import LoadProfile, LoadTestConfig
from asr_assess.core.load_results import (
    GaugeStats,
    GpuWindow,
    LevelResult,
    LoadRunSummary,
    Verdict,
)
from asr_assess.core.manifest import ManifestEntry
from asr_assess.core.run_record import RunRecord
from asr_assess.core.transcription_api import encode_wav

log = logging.getLogger(__name__)

Clock = Callable[[], float]
Window = tuple[float, float]


@dataclass(frozen=True)
class RunMeta:
    """Provenance stamped on a run: label, target, server config text, code and hardware."""

    label: str
    url: str
    server_config: str | None
    record: RunRecord
    env: EnvInfo


def build_pool(entries: Sequence[ManifestEntry], sample_rate: int) -> list[PoolClip]:
    """Load every clip once and encode it as the WAV bytes that will be sent."""
    return [
        PoolClip(
            id=e.audio,
            audio_s=e.duration,
            bucket=e.bucket,
            reference=e.transcript,
            wav=encode_wav(load_audio(Path(e.audio), sample_rate), sample_rate),
        )
        for e in entries
    ]


async def run_stream(
    transport: Transport,
    clips: Sequence[PoolClip],
    order: Sequence[int],
    ids: tuple[int, int, int],
    stop_at: float,
    clock: Clock,
) -> list[RequestRecord]:
    """One closed-loop client; ``ids`` is (stream, level, repeat). Stops sending at ``stop_at``."""
    stream, level, repeat = ids
    records: list[RequestRecord] = []
    i = 0
    while clock() < stop_at:
        clip = clips[order[i % len(order)]]
        i += 1
        sent = clock()
        result = await transport.transcribe(clip.wav)
        records.append(
            RequestRecord(
                clip.id,
                stream,
                level,
                repeat,
                clip.audio_s,
                clip.bucket,
                sent,
                clock(),
                result.status,
                result.text,
            )
        )
    return records


async def run_level(
    transport: Transport,
    clips: Sequence[PoolClip],
    level: int,
    repeat: int,
    profile: LoadProfile,
    seed: int,
    clock: Clock,
) -> tuple[list[RequestRecord], Window]:
    """``level`` streams through warm-up and steady state; the steady-state requests and window."""
    start = clock() + profile.warmup_s
    end = start + profile.steady_state_s
    streams = [
        run_stream(
            transport, clips, stream_order(len(clips), s, seed), (s, level, repeat), end, clock
        )
        for s in range(level)
    ]
    records = [r for stream in await asyncio.gather(*streams) for r in stream]
    return in_window(records, start, end), (start, end)


@dataclass
class _Run:
    """State shared by the levels of one run."""

    transport: Transport
    clips: Sequence[PoolClip]
    cfg: LoadTestConfig
    profile: LoadProfile
    out_dir: Path
    gpu: GpuMonitor | None
    clock: Clock
    scrapes: list[Scrape] = field(default_factory=list)
    results: dict[int, LevelResult] = field(default_factory=dict)

    async def measure(self, level: int) -> LevelResult:
        """Run ``level`` for every repeat, write its requests and summary, and keep the result."""
        records: list[RequestRecord] = []
        windows: list[Window] = []
        for repeat in range(self.profile.repeats):
            got, window = await run_level(
                self.transport,
                self.clips,
                level,
                repeat,
                self.profile,
                self.cfg.audio_pool.seed,
                self.clock,
            )
            records += got
            windows.append(window)
        result = summarize_level(
            level,
            records,
            {c.id: c.reference for c in self.clips},
            self.profile.steady_state_s,
            self.profile.repeats,
            self._gpu_window(windows),
            self._vllm_window(windows),
        )
        self._append(records, result)
        self.results[level] = result
        log.info("level %d: %s", level, _brief(result))
        return result

    def passes(self, level: int, threshold: float) -> bool:
        """Whether a measured level is sustainable at ``threshold``."""
        return is_sustainable(
            self.results[level],
            threshold,
            self.reference_wer(),
            self.cfg.thresholds.max_wer_delta_points,
        )

    def reference_wer(self) -> float | None:
        """The level-1 WER of this run."""
        first = self.results.get(1)
        return first.wer if first else None

    def _gpu_window(self, windows: Sequence[Window]) -> GpuWindow | None:
        if self.gpu is None:
            return None
        inside = [s for s in self.gpu.samples() if _within(s.t, windows)]
        return summarize_gpu(inside, -math.inf, math.inf)

    def _vllm_window(self, windows: Sequence[Window]) -> dict[str, GaugeStats] | None:
        inside = [s for s in self.scrapes if _within(s[0], windows)]
        return summarize_gauges(inside, -math.inf, math.inf) or None

    def _append(self, records: Sequence[RequestRecord], result: LevelResult) -> None:
        with (self.out_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        with (self.out_dir / "levels.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")


async def run_loadtest(
    transport: Transport,
    clips: Sequence[PoolClip],
    cfg: LoadTestConfig,
    profile_name: str,
    out_dir: Path,
    meta: RunMeta,
    gpu: GpuMonitor | None,
    clock: Clock = time.monotonic,
) -> LoadRunSummary:
    """Sweep the profile's levels, bisect near the limit, and write the run's result files."""
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    if not await transport.wait_ready(cfg.request_timeout_s):
        raise RuntimeError(f"Server not ready at {meta.url}")
    out_dir.mkdir(parents=True)
    run = _Run(transport, clips, cfg, cfg.profiles[profile_name], out_dir, gpu, clock)
    scraper = asyncio.create_task(_scrape(transport, 1.0 / cfg.metrics_scrape_hz, clock, run))
    try:
        await _sweep(run)
        await _bisect(run)
    finally:
        scraper.cancel()
        await asyncio.gather(scraper, return_exceptions=True)
    summary = _summary(run, profile_name, meta)
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", "utf-8")
    return summary


async def _sweep(run: _Run) -> None:
    """Levels in order; after the first failure, ``stop_after_failures`` more, then stop."""
    extra: int | None = None
    for level in run.profile.concurrency_levels:
        if extra is not None:
            if extra == 0:
                return
            extra -= 1
        await run.measure(level)
        if extra is None and not run.passes(level, run.cfg.thresholds.p95_rtf_max):
            extra = run.profile.stop_after_failures


async def _bisect(run: _Run) -> None:
    """Binary search between the last passing and the first failing level at P95 <= max."""
    if not run.profile.bisect:
        return
    threshold = run.cfg.thresholds.p95_rtf_max
    tested = sorted(run.results)
    failing = next((lv for lv in tested if not run.passes(lv, threshold)), None)
    passing = max((lv for lv in tested if failing and lv < failing), default=None)
    if failing is None or passing is None or not run.passes(passing, threshold):
        return
    for _ in range(run.profile.bisect_max_steps):
        mid = bisect_next(passing, failing)
        if mid is None:
            return
        await run.measure(mid)
        passing, failing = (mid, failing) if run.passes(mid, threshold) else (passing, mid)


async def _scrape(transport: Transport, period_s: float, clock: Clock, run: _Run) -> None:
    """Scrape server gauges until cancelled; stop at once if the server has no metrics."""
    while True:
        text = await transport.scrape_metrics()
        if text is None:
            return
        run.scrapes.append((clock(), parse_gauges(text)))
        await asyncio.sleep(period_s)


def _summary(run: _Run, profile_name: str, meta: RunMeta) -> LoadRunSummary:
    thresholds = (run.cfg.thresholds.p95_rtf_max, run.cfg.thresholds.p95_rtf_strong)
    verdicts = [
        Verdict(
            threshold=t,
            max_sustainable=max_sustainable({lv: run.passes(lv, t) for lv in run.results}),
        )
        for t in thresholds
    ]
    return LoadRunSummary(
        label=meta.label,
        profile=profile_name,
        url=meta.url,
        server_config=meta.server_config,
        record=meta.record.model_dump(mode="json"),
        env=asdict(meta.env),
        reference_wer=run.reference_wer(),
        levels=[run.results[lv] for lv in sorted(run.results)],
        verdicts=verdicts,
    )


def _within(t: float, windows: Sequence[Window]) -> bool:
    return any(start <= t < end for start, end in windows)


def _brief(result: LevelResult) -> str:
    p95 = f"{result.rtf.p95:.3f}" if result.rtf else "n/a"
    return f"P95 RTF {p95}, ok {result.n_ok}/{result.n_sent}, WER {result.wer}"
