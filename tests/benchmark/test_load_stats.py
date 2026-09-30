"""Tests for the pure load-test maths."""

from pathlib import Path

import pytest

from asr_assess.benchmark.load_stats import (
    RequestRecord,
    bisect_next,
    in_window,
    is_sustainable,
    max_sustainable,
    stream_order,
    summarize_level,
)
from asr_assess.core.load_results import LoadRunSummary, Verdict, read_summary

REFS = {"c1": "a b"}


def rec(
    sent: float, latency: float, status: str = "ok", text: str = "a b", audio: float = 2.0
) -> RequestRecord:
    return RequestRecord("c1", 0, 1, 0, audio, "2-5", sent, sent + latency, status, text)


def test_stream_order_is_seeded_and_differs_per_stream() -> None:
    assert stream_order(10, 0, 7) == stream_order(10, 0, 7)
    assert stream_order(10, 0, 7) != stream_order(10, 1, 7)
    assert sorted(stream_order(10, 3, 7)) == list(range(10))


def test_window_selects_by_send_time_keeping_slow_finishers() -> None:
    records = [rec(0.5, 0.1), rec(1.0, 9.0), rec(2.9, 5.0), rec(3.0, 0.1)]
    assert [r.sent for r in in_window(records, 1.0, 3.0)] == [1.0, 2.9]


def test_summarize_level_rtf_throughput_and_wer() -> None:
    records = [rec(0, 0.2), rec(1, 0.4, text="a x")]
    res = summarize_level(1, records, REFS, window_s=2.0, repeats=1, gpu=None, vllm=None)
    assert res.rtf is not None
    assert res.rtf.max == pytest.approx(0.2)  # 0.4 s for 2 s of audio
    assert res.audio_s_per_s == 2.0
    assert res.requests_per_s == 1.0
    assert res.wer == 0.25
    assert (res.n_sent, res.n_ok) == (2, 2)


def test_all_failed_level_summarizes_and_fails() -> None:
    res = summarize_level(8, [rec(0, 60, status="timeout")], REFS, 2.0, 1, None, None)
    assert res.rtf is None
    assert res.wer is None
    assert res.n_timeout == 1
    assert not is_sustainable(res, 0.5, reference_wer=0.1, max_wer_delta_points=1.0)


def test_each_condition_fails_on_its_own() -> None:
    ok = summarize_level(1, [rec(0, 0.2)], REFS, 2.0, 1, None, None)
    assert is_sustainable(ok, 0.5, 0.0, 1.0)
    assert not is_sustainable(ok, 0.05, 0.0, 1.0)  # P95 over threshold
    errors = summarize_level(
        1, [rec(0, 0.2), rec(1, 0.1, status="http_500")], REFS, 2.0, 1, None, None
    )
    assert errors.n_error == 1
    assert not is_sustainable(errors, 0.5, 0.0, 1.0)  # any failed request
    worse = summarize_level(1, [rec(0, 0.2, text="x y")], REFS, 2.0, 1, None, None)
    assert not is_sustainable(worse, 0.5, 0.0, 1.0)  # WER +100 points


def test_wer_within_the_allowed_delta_passes() -> None:
    res = summarize_level(1, [rec(0, 0.2, text="a x")], REFS, 2.0, 1, None, None)  # WER 0.5
    assert is_sustainable(res, 0.5, reference_wer=0.495, max_wer_delta_points=1.0)


def test_bisection_and_max() -> None:
    assert bisect_next(4, 8) == 6
    assert bisect_next(5, 6) is None
    assert max_sustainable({1: True, 2: True, 4: False, 3: True}) == 3
    assert max_sustainable({1: False}) is None


def test_a_fluke_pass_above_a_failure_does_not_count() -> None:
    assert max_sustainable({1: True, 2: False, 4: True}) == 1


def test_summary_round_trips_through_json(tmp_path: Path) -> None:
    level = summarize_level(1, [rec(0, 0.2)], REFS, 2.0, 1, None, None)
    summary = LoadRunSummary(
        label="x",
        profile="quick",
        url="http://s",
        server_config=None,
        record={},
        env={},
        reference_wer=0.0,
        levels=[level],
        verdicts=[Verdict(threshold=0.5, max_sustainable=1)],
    )
    path = tmp_path / "summary.json"
    path.write_text(summary.model_dump_json(), encoding="utf-8")
    assert read_summary(path) == summary
