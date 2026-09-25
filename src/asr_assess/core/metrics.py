"""Pure evaluation maths: WER/CER, bootstrap confidence intervals, RTF and throughput."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jiwer
import numpy as np

Unit = Literal["word", "char"]


@dataclass(frozen=True)
class ErrorCounts:
    """Edit errors (substitutions + deletions + insertions) against a reference length."""

    errors: int
    ref_len: int


@dataclass(frozen=True)
class RTFStats:
    """Distribution summary of real-time factors."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


@dataclass(frozen=True)
class Throughput:
    """Served audio seconds and requests per wall-clock second."""

    audio_s_per_s: float
    requests_per_s: float


def utterance_errors(reference: str, hypothesis: str, unit: Unit) -> ErrorCounts:
    """Count edit errors of one already-normalized hypothesis against its reference."""
    ref_tokens, hyp_tokens = _tokens(reference, unit), _tokens(hypothesis, unit)
    if not ref_tokens or not hyp_tokens:
        return ErrorCounts(max(len(ref_tokens), len(hyp_tokens)), len(ref_tokens))
    out: jiwer.WordOutput | jiwer.CharacterOutput
    if unit == "word":
        out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    else:
        out = jiwer.process_characters("".join(ref_tokens), "".join(hyp_tokens))
    return ErrorCounts(out.substitutions + out.deletions + out.insertions, len(ref_tokens))


def corpus_rate(counts: Sequence[ErrorCounts]) -> float:
    """Corpus-level error rate: total errors over total reference length."""
    ref_total = sum(c.ref_len for c in counts)
    if ref_total <= 0:
        raise ValueError("Total reference length must be positive")
    return sum(c.errors for c in counts) / ref_total


def bootstrap_ci(
    counts: Sequence[ErrorCounts], n_resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap interval of the corpus error rate, resampling utterances."""
    if not counts:
        raise ValueError("Need at least one utterance")
    errors = np.array([c.errors for c in counts], dtype=np.float64)
    lengths = np.array([c.ref_len for c in counts], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_resamples, len(counts)))
    ref_sums = lengths[idx].sum(axis=1)
    valid = ref_sums > 0
    rates = errors[idx].sum(axis=1)[valid] / ref_sums[valid]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(rates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def rtf(processing_s: float, audio_s: float) -> float:
    """Real-time factor: processing time divided by audio duration."""
    if audio_s <= 0:
        raise ValueError("Audio duration must be positive")
    return processing_s / audio_s


def summarize_rtf(values: Sequence[float]) -> RTFStats:
    """Mean, P50, P95, P99 and max of per-request RTFs (linear-interpolated percentiles)."""
    if not values:
        raise ValueError("Need at least one RTF value")
    arr = np.asarray(values, dtype=np.float64)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return RTFStats(
        len(arr), float(arr.mean()), float(p50), float(p95), float(p99), float(arr.max())
    )


def throughput(total_audio_s: float, n_requests: int, wall_s: float) -> Throughput:
    """Audio seconds and requests completed per wall-clock second."""
    if wall_s <= 0:
        raise ValueError("Wall time must be positive")
    return Throughput(total_audio_s / wall_s, n_requests / wall_s)


def _tokens(text: str, unit: Unit) -> list[str]:
    words = text.split()
    return words if unit == "word" else list("".join(words))
