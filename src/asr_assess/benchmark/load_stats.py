"""Pure load-test maths: stream and speaker schedules, the window, per-level stats, verdicts."""

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
    """One request as the client saw it; RTF = (finished - sent) / audio_s.

    ``due``: open loop only, when the speaker finished the utterance (the scheduled send time).
    """

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
    due: float | None = None


def stream_order(n_clips: int, stream: int, seed: int) -> list[int]:
    """The seeded order in which ``stream`` walks the pool (a permutation, cycled)."""
    return random.Random(seed + stream).sample(range(n_clips), n_clips)


def session_phase(first_audio_s: float, stream: int, seed: int) -> float:
    """When ``stream``'s first utterance is sent, in seconds after the level starts.

    Each live speaker is caught a uniform-random way through its first utterance, so N sessions
    send at a steady rate from the start instead of all at once one clip later.
    """
    return random.Random(f"{seed}:{stream}:phase").uniform(0.0, first_audio_s)


def open_schedule(
    audio_s: Sequence[float],
    order: Sequence[int],
    first_send: float,
    pause_s: float,
    stop_at: float,
) -> list[tuple[float, int]]:
    """One live speaker's ``(send time, clip index)`` for every send before ``stop_at``.

    After each send the speaker pauses ``pause_s``, then speaks the next clip of ``order``
    (cycled) and sends it as soon as it has been said: it never waits for a transcript.
    """
    if pause_s <= 0.0 and min(audio_s[i] for i in order) <= 0.0:
        raise ValueError("a zero-length clip with no pause would schedule sends forever")
    schedule: list[tuple[float, int]] = []
    at, k = first_send, 0
    while at < stop_at:
        schedule.append((at, order[k % len(order)]))
        k += 1
        at += pause_s + audio_s[order[k % len(order)]]
    return schedule


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
    lags = [r.sent - r.due for r in records if r.due is not None]
    lag = summarize_rtf(lags) if lags else None
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
        client_lag_p95_s=lag.p95 if lag else None,
        client_lag_max_s=lag.max if lag else None,
    )


def is_sustainable(
    result: LevelResult,
    threshold: float,
    reference_wer: float | None,
    max_wer_delta_points: float,
    max_client_lag_s: float | None = None,
) -> bool:
    """P95 RTF within ``threshold``, no failed request, WER within the allowed delta, and (open
    loop) the client on its timetable: a lagging client offered less load than the level claims.
    """
    if result.rtf is None or result.rtf.p95 > threshold:
        return False
    if result.n_timeout or result.n_error:
        return False
    lag = result.client_lag_p95_s
    if max_client_lag_s is not None and lag is not None and lag > max_client_lag_s:
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
