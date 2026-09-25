"""Tests for per-utterance and grouped WER/CER scoring."""

import pytest

from asr_assess.core.config import BootstrapSettings
from asr_assess.core.manifest import ManifestEntry
from asr_assess.evaluation.scoring import (
    build_report,
    category_of,
    score_group,
    score_utterance,
)
from asr_assess.inference.engine import Transcript

BOOT = BootstrapSettings(n_resamples=200, confidence=0.95, seed=0)


def entry(name: str, text: str, bucket: str = "2-5", source: str = "manglish/src") -> ManifestEntry:
    return ManifestEntry(
        audio=f"{name}.wav",
        text=f"language None<asr_text>{text}",
        duration=3.0,
        bucket=bucket,
        source=source,
    )


def test_category_is_the_source_prefix() -> None:
    assert category_of("malay_read/fleurs_ms_test") == "malay_read"


def test_normalization_is_applied_to_both_sides() -> None:
    item = score_utterance(entry("a", "Hello, World!"), Transcript("a.wav", "hello world", None))
    assert item.words.errors == 0 and item.chars.errors == 0


def test_empty_hypothesis_counts_every_reference_word() -> None:
    item = score_utterance(entry("a", "satu dua tiga"), Transcript("a.wav", "", None))
    assert (item.words.errors, item.words.ref_len) == (3, 3)


def test_mismatched_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="id"):
        score_utterance(entry("a", "x y"), Transcript("b.wav", "x y", None))


def test_perfect_group_has_zero_wer_and_degenerate_ci() -> None:
    items = [
        score_utterance(entry(n, "satu dua"), Transcript(f"{n}.wav", "satu dua", None))
        for n in "abc"
    ]
    group = score_group(items, BOOT)
    assert group.n == 3
    assert (group.wer.value, group.wer.low, group.wer.high) == (0.0, 0.0, 0.0)


def test_group_ci_contains_the_point_estimate() -> None:
    pairs = [("satu dua tiga", "satu dua tiga"), ("empat lima", "empat"), ("enam", "tujuh")]
    items = [
        score_utterance(entry(str(i), r), Transcript(f"{i}.wav", h, None))
        for i, (r, h) in enumerate(pairs)
    ]
    wer = score_group(items, BOOT).wer
    assert wer.value is not None and wer.low is not None and wer.high is not None
    assert wer.low <= wer.value <= wer.high


def test_group_with_only_empty_references_reports_none() -> None:
    item = score_utterance(entry("a", "!!!"), Transcript("a.wav", "", None))
    assert score_group([item], BOOT).wer == score_group([item], BOOT).cer
    assert score_group([item], BOOT).wer.value is None


def test_report_groups_by_bucket_and_category() -> None:
    items = [
        score_utterance(entry("a", "x y", "2-5", "manglish/s"), Transcript("a.wav", "x y", None)),
        score_utterance(entry("b", "x y", "5-15", "malay_read/s"), Transcript("b.wav", "x", None)),
    ]
    report = build_report(items, BOOT)
    assert report.overall.n == 2
    assert set(report.by_bucket) == {"2-5", "5-15"}
    assert set(report.by_category) == {"manglish", "malay_read"}
    assert report.by_category["malay_read"].wer.value == pytest.approx(0.5)
