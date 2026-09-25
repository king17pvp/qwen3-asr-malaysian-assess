"""Tests for WER/CER, bootstrap confidence intervals, RTF percentiles and throughput."""

import pytest

from asr_assess.core.metrics import (
    ErrorCounts,
    bootstrap_ci,
    corpus_rate,
    rtf,
    summarize_rtf,
    throughput,
    utterance_errors,
)


class TestUtteranceErrors:
    def test_word_errors_count_sub_del_ins(self) -> None:
        # "a b c" -> "a x c d": one substitution, one insertion.
        assert utterance_errors("a b c", "a x c d", unit="word") == ErrorCounts(2, 3)

    def test_char_errors(self) -> None:
        assert utterance_errors("abc", "abd", unit="char") == ErrorCounts(1, 3)

    def test_char_errors_ignore_spaces(self) -> None:
        assert utterance_errors("ab c", "abc", unit="char") == ErrorCounts(0, 3)

    def test_empty_reference_counts_insertions(self) -> None:
        assert utterance_errors("", "a b", unit="word") == ErrorCounts(2, 0)

    def test_both_empty_is_zero(self) -> None:
        assert utterance_errors("", "", unit="word") == ErrorCounts(0, 0)


class TestCorpusRate:
    def test_is_total_errors_over_total_reference_length(self) -> None:
        counts = [ErrorCounts(1, 4), ErrorCounts(0, 6)]
        assert corpus_rate(counts) == pytest.approx(0.1)

    def test_rejects_zero_reference_length(self) -> None:
        with pytest.raises(ValueError):
            corpus_rate([ErrorCounts(1, 0)])


class TestBootstrapCI:
    def test_interval_contains_point_estimate(self) -> None:
        counts = [ErrorCounts(e, 10) for e in (0, 1, 2, 3, 4, 5, 1, 2, 0, 3)]
        low, high = bootstrap_ci(counts, n_resamples=500, confidence=0.95, seed=0)
        assert low <= corpus_rate(counts) <= high
        assert low < high

    def test_same_seed_is_deterministic(self) -> None:
        counts = [ErrorCounts(e, 5) for e in (0, 1, 2, 3)]
        first = bootstrap_ci(counts, n_resamples=200, confidence=0.9, seed=7)
        assert bootstrap_ci(counts, n_resamples=200, confidence=0.9, seed=7) == first

    def test_constant_errors_give_degenerate_interval(self) -> None:
        counts = [ErrorCounts(1, 10)] * 5
        assert bootstrap_ci(counts, n_resamples=100, confidence=0.95, seed=0) == pytest.approx(
            (0.1, 0.1)
        )


class TestRTF:
    def test_rtf_is_processing_over_audio(self) -> None:
        assert rtf(processing_s=1.5, audio_s=6.0) == pytest.approx(0.25)

    def test_rtf_rejects_nonpositive_audio(self) -> None:
        with pytest.raises(ValueError):
            rtf(processing_s=1.0, audio_s=0.0)

    def test_summary_percentiles(self) -> None:
        stats = summarize_rtf([float(v) for v in range(1, 101)])
        assert stats.count == 100
        assert stats.mean == pytest.approx(50.5)
        assert stats.p50 == pytest.approx(50.5)
        assert stats.p95 == pytest.approx(95.05)
        assert stats.p99 == pytest.approx(99.01)
        assert stats.max == 100.0

    def test_summary_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            summarize_rtf([])


class TestThroughput:
    def test_audio_and_request_rates(self) -> None:
        result = throughput(total_audio_s=300.0, n_requests=40, wall_s=60.0)
        assert result.audio_s_per_s == pytest.approx(5.0)
        assert result.requests_per_s == pytest.approx(40 / 60)

    def test_rejects_nonpositive_wall_time(self) -> None:
        with pytest.raises(ValueError):
            throughput(total_audio_s=1.0, n_requests=1, wall_s=0.0)
