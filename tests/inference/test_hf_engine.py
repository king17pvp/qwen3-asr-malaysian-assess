"""Tests for the Transformers engine's pure helpers (no torch needed)."""

from asr_assess.inference.hf_engine import hit_token_limit, language_argument


def test_no_hints_means_auto_detect() -> None:
    assert language_argument([None, None]) is None


def test_one_shared_hint_is_passed_once() -> None:
    assert language_argument(["Malay", "Malay"]) == "Malay"


def test_mixed_hints_are_passed_per_clip() -> None:
    assert language_argument(["Malay", None]) == ["Malay", None]


def test_token_limit_detection() -> None:
    assert hit_token_limit(256, 256)
    assert not hit_token_limit(40, 256)
