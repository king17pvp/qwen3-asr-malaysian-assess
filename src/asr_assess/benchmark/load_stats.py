"""Pure load-test maths: stream schedules, the measurement window, per-level stats, verdicts."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from asr_assess.core.load_results import GaugeStats, GpuWindow, LevelResult, RTFSummary
from asr_assess.core.metrics import (
    ErrorCounts,
    Unit,
    corpus_rate,
    rtf,
    summarize_rtf,
    throughput,
    utterance_errors,
)
from asr_assess.core.text import normalize_text

_POINTS = 100.0  # WER is a fraction; the threshold is in percentage points


@dataclass(frozen=True)
class PoolClip:
    """One utterance of the audio pool, already encoded as the WAV bytes that get sent."""

    id: str
    audio_s: float
    bucket: str
    reference: str
    wav: bytes


@dataclass(frozen=True)
class RequestRecord:
    """One request as the client saw it; RTF = (finished - sent) / audio_s."""

    id: str
    stream: int
    level: int
    repeat: int
    audio_s: float
    bucket: str
    sent: float
    finished: float
    status: str
    text: str


def stream_order(n_clips: int, stream: int, seed: int) -> list[int]:
    """The seeded order in which ``stream`` walks the pool (a permutation, cycled)."""
    return random.Random(seed + stream).sample(range(n_clips), n_clips)


def in_window(records: Sequence[RequestRecord], start: float, end: float) -> list[RequestRecord]:
    """Requests *sent* in ``[start, end)``: slow ones finishing later still count."""
    return [r for r in records if start <= r.sent < end]


def summarize_level(
    level: int,
    records: Sequence[RequestRecord],
    references: Mapping[str, str],
    window_s: float,
    repeats: int,
    gpu: GpuWindow | None,
    vllm: dict[str, GaugeStats] | None,
) -> LevelResult:
    """RTF, throughput and WER over successful requests; failures are counted, not timed."""
    ok = [r for r in records if r.status == "ok"]
    n_timeout = sum(r.status == "timeout" for r in records)
    rates = throughput(sum(r.audio_s for r in ok), len(ok), window_s * repeats)
    stats = summarize_rtf([rtf(r.finished - r.sent, r.audio_s) for r in ok]) if ok else None
    return LevelResult(
        level=level,
        repeats=repeats,
        n_sent=len(records),
        n_ok=len(ok),
        n_timeout=n_timeout,
        n_error=len(records) - len(ok) - n_timeout,
        rtf=RTFSummary(**asdict(stats)) if stats else None,
        requests_per_s=rates.requests_per_s,
        audio_s_per_s=rates.audio_s_per_s,
        wer=_error_rate(ok, references, "word"),
        cer=_error_rate(ok, references, "char"),
        gpu=gpu,
        vllm=vllm,
    )


def is_sustainable(
    result: LevelResult, threshold: float, reference_wer: float | None, max_wer_delta_points: float
) -> bool:
    """P95 RTF within ``threshold``, no failed request, and WER within the allowed delta."""
    if result.rtf is None or result.rtf.p95 > threshold:
        return False
    if result.n_timeout or result.n_error:
        return False
    if reference_wer is None or result.wer is None:
        return True
    return (result.wer - reference_wer) * _POINTS <= max_wer_delta_points


def bisect_next(passing: int, failing: int) -> int | None:
    """The midpoint strictly between a passing and a failing level, or None if adjacent."""
    mid = (passing + failing) // 2
    return mid if passing < mid < failing else None


def max_sustainable(verdicts: Mapping[int, bool]) -> int | None:
    """Highest level such that it and every lower tested level passed."""
    best = None
    for level in sorted(verdicts):
        if not verdicts[level]:
            break
        best = level
    return best


def _error_rate(
    ok: Sequence[RequestRecord], references: Mapping[str, str], unit: Unit
) -> float | None:
    counts: list[ErrorCounts] = [
        utterance_errors(normalize_text(references[r.id]), normalize_text(r.text), unit) for r in ok
    ]
    if sum(c.ref_len for c in counts) == 0:
        return None
    return corpus_rate(counts)
