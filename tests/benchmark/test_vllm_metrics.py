"""Tests for parsing vLLM's Prometheus scheduler gauges."""

from pathlib import Path

from asr_assess.benchmark.vllm_metrics import parse_gauges, summarize_gauges

SAMPLE = (Path(__file__).parent / "data" / "vllm_metrics_sample.txt").read_text(encoding="utf-8")


def test_parses_running_waiting_and_kv() -> None:
    assert parse_gauges(SAMPLE) == {"running": 3.0, "waiting": 1.0, "kv_cache_usage": 0.42}


def test_sums_label_sets_and_ignores_comments() -> None:
    text = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{model_name="a"} 2.0\n'
        'vllm:num_requests_running{model_name="b"} 3.0\n'
    )
    assert parse_gauges(text) == {"running": 5.0}


def test_older_kv_gauge_name_is_accepted() -> None:
    assert parse_gauges("vllm:gpu_cache_usage_perc 0.5\n") == {"kv_cache_usage": 0.5}


def test_summarize_uses_only_the_window() -> None:
    scrapes = [(0.0, {"running": 9.0}), (1.0, {"running": 2.0}), (2.0, {"running": 4.0})]
    stats = summarize_gauges(scrapes, 1.0, 3.0)
    assert (stats["running"].mean, stats["running"].max) == (3.0, 4.0)


def test_summarize_empty_window_is_empty() -> None:
    assert summarize_gauges([(0.0, {"running": 1.0})], 5.0, 6.0) == {}
