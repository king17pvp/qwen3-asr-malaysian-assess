"""Tests for the pure load-test maths."""

import json
from pathlib import Path

import pytest

from asr_assess.benchmark.load_stats import (
    RequestRecord,
    bisect_next,
    in_window,
    is_sustainable,
    max_sustainable,
    open_schedule,
    session_phase,
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


def test_open_schedule_sends_each_clip_once_spoken() -> None:
    audio = [2.0, 3.0, 5.0]
    schedule = open_schedule(audio, [2, 0, 1], first_send=1.0, pause_s=0.5, stop_at=14.0)
    # 1.0 clip 2 | +0.5 pause +2.0 clip 0 = 3.5 | +0.5 +3.0 = 7.0 | +0.5 +5.0 = 12.5 | next 15.0
    assert schedule == [(1.0, 2), (3.5, 0), (7.0, 1), (12.5, 2)]


def test_open_schedule_cycles_the_order() -> None:
    schedule = open_schedule([1.0, 1.0], [1, 0], first_send=0.0, pause_s=0.0, stop_at=4.0)
    assert schedule == [(0.0, 1), (1.0, 0), (2.0, 1), (3.0, 0)]


def test_open_schedule_is_empty_when_the_first_send_is_too_late() -> None:
    assert open_schedule([2.0], [0], first_send=5.0, pause_s=0.0, stop_at=5.0) == []


def test_open_schedule_rejects_a_schedule_that_never_advances() -> None:
    with pytest.raises(ValueError, match="forever"):
        open_schedule([0.0, 2.0], [0, 1], first_send=0.0, pause_s=0.0, stop_at=1.0)


def test_session_phase_is_seeded_spread_and_within_the_first_clip() -> None:
    phases = [session_phase(6.0, stream, seed=7) for stream in range(200)]
    assert all(0.0 <= p <= 6.0 for p in phases)
    assert phases == [session_phase(6.0, stream, seed=7) for stream in range(200)]
    assert len({round(p, 6) for p in phases}) > 150  # spread out, not one shared offset
    assert session_phase(6.0, 0, seed=8) != phases[0]


def test_summaries_written_before_open_loop_load_as_closed(tmp_path: Path) -> None:
    raw: dict[str, object] = {
        "label": "old",
        "profile": "full",
        "url": "http://s",
        "server_config": None,
        "record": {},
        "env": {},
        "reference_wer": None,
        "levels": [],
        "verdicts": [],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    old = read_summary(path)
    assert (old.mode, old.pause_s) == ("closed", 0.0)


def test_summarize_level_reports_open_loop_client_lag() -> None:
    # sent - due: how late the client sent each request after the speaker finished speaking
    records = [
        RequestRecord("c1", 0, 1, 0, 2.0, "2-5", sent, sent + 0.2, status, "a b", due=sent - lag)
        for sent, lag, status in [(0.0, 0.0, "ok"), (1.0, 0.01, "ok"), (2.0, 0.1, "timeout")]
    ]
    res = summarize_level(1, records, REFS, 2.0, 1, None, None)
    assert res.client_lag_max_s == pytest.approx(0.1)  # failed requests count: lag is client-side
    assert res.client_lag_p95_s == pytest.approx(0.091)


def test_closed_loop_records_have_no_client_lag() -> None:
    res = summarize_level(1, [rec(0, 0.2)], REFS, 2.0, 1, None, None)
    assert (res.client_lag_p95_s, res.client_lag_max_s) == (None, None)


def test_a_level_the_client_fell_behind_on_is_not_sustainable() -> None:
    # the speakers sent late, so the level never offered its load: it must not pass
    lagging = summarize_level(1, [rec(0, 0.2)], REFS, 2.0, 1, None, None).model_copy(
        update={"client_lag_p95_s": 0.8}
    )
    assert not is_sustainable(lagging, 0.5, 0.0, 1.0, max_client_lag_s=0.5)
    on_time = lagging.model_copy(update={"client_lag_p95_s": 0.01})
    assert is_sustainable(on_time, 0.5, 0.0, 1.0, max_client_lag_s=0.5)
