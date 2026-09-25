"""Tests for the Transformers engine's pure helpers (no torch needed)."""

from asr_assess.inference.hf_engine import hit_token_limit, language_argument


def test_no_hints_means_auto_detect() -> None:
    assert language_argument([None, None]) is None


def test_one_shared_hint_is_passed_once() -> None:
    assert language_argument(["Malay", "Malay"]) == "Malay"


def test_mixed_hints_are_passed_per_clip() -> None:
    assert language_argument(["Malay", None]) == ["Malay", None]


EOS = {151645, 151643}


def test_row_that_fills_the_limit_without_eos_is_truncated() -> None:
    assert hit_token_limit([7, 8, 9, 10], EOS, max_new_tokens=4)


def test_row_that_ends_with_eos_is_not_truncated() -> None:
    assert not hit_token_limit([7, 8, 9, 151645], EOS, max_new_tokens=4)


def test_early_finisher_padded_in_a_batch_is_not_truncated() -> None:
    # generate pads rows that stop early with an EOS-like pad id up to the batch's longest row
    assert not hit_token_limit([7, 151645, 151643, 151643], EOS, max_new_tokens=4)


def test_short_row_is_not_truncated() -> None:
    assert not hit_token_limit([7, 8], EOS, max_new_tokens=4)
