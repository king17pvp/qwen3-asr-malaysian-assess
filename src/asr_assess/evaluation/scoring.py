"""Pure WER/CER scoring: per utterance, then per group with bootstrap confidence intervals."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from asr_assess.core.config import BootstrapSettings
from asr_assess.core.manifest import ManifestEntry
from asr_assess.core.metrics import ErrorCounts, bootstrap_ci, corpus_rate, utterance_errors
from asr_assess.core.text import normalize_text
from asr_assess.inference.engine import Transcript


@dataclass(frozen=True)
class Scored:
    """One utterance with its normalized texts and edit-error counts."""

    id: str
    source: str
    category: str
    bucket: str
    duration: float
    reference: str  # normalized, as scored
    hypothesis: str  # normalized, as scored
    reference_raw: str  # manifest transcript, so results can be re-scored without a GPU
    hypothesis_raw: str  # engine output
    detected_language: str | None
    hit_token_limit: bool
    words: ErrorCounts
    chars: ErrorCounts


@dataclass(frozen=True)
class RateCI:
    """An error rate and its confidence interval; all None when there is no reference text."""

    value: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class GroupScore:
    """Corpus WER/CER over a group of utterances."""

    n: int
    audio_s: float
    wer: RateCI
    cer: RateCI


@dataclass(frozen=True)
class EvalReport:
    """Scores overall, per duration bucket and per data category."""

    overall: GroupScore
    by_bucket: dict[str, GroupScore]
    by_category: dict[str, GroupScore]


def category_of(source: str) -> str:
    """Data category of a manifest ``source`` such as ``malay_read/fleurs_ms_test``."""
    return source.split("/", 1)[0]


def score_utterance(entry: ManifestEntry, transcript: Transcript) -> Scored:
    """Normalize reference and hypothesis, then count word and character errors."""
    if transcript.id != entry.audio:
        raise ValueError(f"Transcript id {transcript.id!r} does not match {entry.audio!r}")
    ref, hyp = normalize_text(entry.transcript), normalize_text(transcript.text)
    return Scored(
        id=entry.audio,
        source=entry.source,
        category=category_of(entry.source),
        bucket=entry.bucket,
        duration=entry.duration,
        reference=ref,
        hypothesis=hyp,
        reference_raw=entry.transcript,
        hypothesis_raw=transcript.text,
        detected_language=transcript.language,
        hit_token_limit=transcript.hit_token_limit,
        words=utterance_errors(ref, hyp, unit="word"),
        chars=utterance_errors(ref, hyp, unit="char"),
    )


def score_group(items: Sequence[Scored], bootstrap: BootstrapSettings) -> GroupScore:
    """Corpus WER and CER of ``items`` with bootstrap intervals."""
    return GroupScore(
        n=len(items),
        audio_s=round(sum(i.duration for i in items), 3),
        wer=_rate([i.words for i in items], bootstrap),
        cer=_rate([i.chars for i in items], bootstrap),
    )


def build_report(items: Sequence[Scored], bootstrap: BootstrapSettings) -> EvalReport:
    """Overall, per-bucket and per-category scores."""
    return EvalReport(
        overall=score_group(items, bootstrap),
        by_bucket=_grouped(items, lambda i: i.bucket, bootstrap),
        by_category=_grouped(items, lambda i: i.category, bootstrap),
    )


def _grouped(
    items: Sequence[Scored], key: Callable[[Scored], str], bootstrap: BootstrapSettings
) -> dict[str, GroupScore]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return {name: score_group(group, bootstrap) for name, group in sorted(groups.items())}


def _rate(counts: Sequence[ErrorCounts], bootstrap: BootstrapSettings) -> RateCI:
    if sum(c.ref_len for c in counts) == 0:
        return RateCI(None, None, None)
    low, high = bootstrap_ci(counts, bootstrap.n_resamples, bootstrap.confidence, bootstrap.seed)
    return RateCI(corpus_rate(counts), low, high)
