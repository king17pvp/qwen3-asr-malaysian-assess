"""Tests for Whisper-token stripping, the WER normalizer and Qwen prefix helpers."""

import pytest

from asr_assess.core.text import (
    build_qwen_target,
    normalize_text,
    parse_qwen_output,
    strip_whisper_tokens,
    whisper_language_code,
)

MESOLITICA_ROW = (
    "<|startoftranscript|><|ms|><|transcribe|><|0.00|> tu lah<|0.28|><|1.12|> Total<|1.52|>"
    "<|2.66|> Kalau macam Singapore<|5.68|><|endoftext|>"
)


class TestStripWhisperTokens:
    def test_removes_special_and_timestamp_tokens(self) -> None:
        assert strip_whisper_tokens(MESOLITICA_ROW) == "tu lah Total Kalau macam Singapore"

    def test_keeps_space_between_words_split_by_timestamps(self) -> None:
        assert strip_whisper_tokens("<|0.00|>satu<|0.40|><|0.50|>dua<|0.90|>") == "satu dua"

    def test_plain_text_is_unchanged(self) -> None:
        assert strip_whisper_tokens("hello world") == "hello world"

    def test_only_tokens_gives_empty_string(self) -> None:
        assert strip_whisper_tokens("<|startoftranscript|><|endoftext|>") == ""


class TestWhisperLanguageCode:
    def test_reads_language_token(self) -> None:
        assert whisper_language_code(MESOLITICA_ROW) == "ms"

    def test_ignores_timestamps_and_task_tokens(self) -> None:
        assert whisper_language_code("<|startoftranscript|><|en|><|transcribeprecise|>") == "en"

    def test_returns_none_without_language_token(self) -> None:
        assert whisper_language_code("<|0.00|> hello<|0.40|>") is None


class TestNormalizeText:
    def test_lowercases_and_removes_punctuation(self) -> None:
        assert normalize_text("Hello, World! How are you?") == "hello world how are you"

    def test_hyphen_becomes_space(self) -> None:
        # Malay reduplication is written both ways; the metric should not care.
        assert normalize_text("kanak-kanak") == normalize_text("kanak kanak")

    def test_apostrophe_is_dropped_inside_words(self) -> None:
        assert normalize_text("Don't") == "dont"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  a \t b\n c  ") == "a b c"

    def test_unicode_is_nfkc_folded(self) -> None:
        assert normalize_text("ｃａｆé") == "café"

    def test_keeps_digits(self) -> None:
        assert normalize_text("RM 25.50") == "rm 25 50"

    def test_is_idempotent(self) -> None:
        once = normalize_text("Ah, makanan... bagi AKU!")
        assert normalize_text(once) == once


class TestQwenPrefix:
    def test_builds_target_with_language(self) -> None:
        assert build_qwen_target("Malay", "apa khabar") == "language Malay<asr_text>apa khabar"

    def test_builds_target_without_language(self) -> None:
        assert build_qwen_target(None, "boleh lah") == "language None<asr_text>boleh lah"

    def test_rejects_unsupported_language(self) -> None:
        with pytest.raises(ValueError, match="Klingon"):
            build_qwen_target("Klingon", "x")

    def test_parse_round_trips_build(self) -> None:
        assert parse_qwen_output(build_qwen_target("English", "a test")) == ("English", "a test")

    def test_parse_language_none(self) -> None:
        assert parse_qwen_output("language None<asr_text> campur lah ") == (None, "campur lah")

    def test_parse_output_without_tag_is_all_text(self) -> None:
        assert parse_qwen_output("  just text ") == (None, "just text")
