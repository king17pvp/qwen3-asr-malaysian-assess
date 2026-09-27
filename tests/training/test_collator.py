"""Tests for training examples and label masking."""

from pathlib import Path

import pytest

from asr_assess.training.collator import (
    IGNORE_INDEX,
    build_conversation,
    load_examples,
    mask_labels,
)
from tests.fakes import write_clips

HEADER = [7, 8, 9]  # stands in for the ids of "<|im_start|>assistant\n"
X = IGNORE_INDEX


class TestMaskLabels:
    def test_keeps_only_tokens_after_the_header(self) -> None:
        ids = [[1, 2, 7, 8, 9, 20, 21, 5]]
        assert mask_labels(ids, [[1] * 8], HEADER) == [[X, X, X, X, X, 20, 21, 5]]

    def test_uses_the_last_header(self) -> None:
        # A header-like sequence earlier in the prompt must not start the reply.
        ids = [[7, 8, 9, 3, 7, 8, 9, 20, 5]]
        assert mask_labels(ids, [[1] * 9], HEADER) == [[X] * 7 + [20, 5]]

    def test_left_padding_is_masked(self) -> None:
        ids = [[0, 0, 1, 7, 8, 9, 20, 5], [1, 2, 3, 7, 8, 9, 20, 5]]
        mask = [[0, 0, 1, 1, 1, 1, 1, 1], [1] * 8]
        assert mask_labels(ids, mask, HEADER) == [[X] * 6 + [20, 5], [X] * 6 + [20, 5]]

    def test_right_padding_is_masked(self) -> None:
        ids = [[1, 7, 8, 9, 20, 5, 0, 0]]
        mask = [[1, 1, 1, 1, 1, 1, 0, 0]]
        assert mask_labels(ids, mask, HEADER) == [[X, X, X, X, 20, 5, X, X]]

    def test_pad_id_equal_to_eos_is_kept_when_attended(self) -> None:
        # The model config's pad id is <|im_end|>; a real EOS must stay a target.
        eos = 5
        ids = [[eos, 7, 8, 9, 20, eos]]
        mask = [[0, 1, 1, 1, 1, 1]]
        assert mask_labels(ids, mask, HEADER) == [[X, X, X, X, 20, eos]]

    def test_raises_when_a_row_has_no_header(self) -> None:
        with pytest.raises(ValueError, match="assistant header"):
            mask_labels([[1, 2, 3]], [[1, 1, 1]], HEADER)


def test_conversation_is_an_audio_turn_and_the_target_reply() -> None:
    conversation = build_conversation("language Malay<asr_text>satu")
    assert conversation == [
        {"role": "user", "content": [{"type": "audio"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "language Malay<asr_text>satu"}],
        },
    ]


def test_load_examples_reads_audio_and_keeps_the_full_target(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, [("a", 2.5, "2-5", "satu dua")])
    [example] = load_examples(entries, 16000)
    assert len(example.samples) == 40000
    assert example.target == "language Malay<asr_text>satu dua"
