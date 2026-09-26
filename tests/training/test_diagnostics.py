"""Tests for trainable-parameter counts and weight-delta summaries."""

import numpy as np
import pytest

from asr_assess.training.diagnostics import count_parameters, relative_delta, summarize_deltas
from asr_assess.training.lora import decoder_target_pattern

PATTERN = decoder_target_pattern(["q_proj", "down_proj"])
Q = "model.language_model.layers.0.self_attn.q_proj.weight"
DOWN = "model.language_model.layers.0.mlp.down_proj.weight"
ENC = "model.audio_tower.layers.0.self_attn.q_proj.weight"
EMB = "model.language_model.embed_tokens.weight"


def test_count_parameters_by_component() -> None:
    p, q = "base_model.model.model.", "language_model.layers.0.self_attn.q_proj"
    count = count_parameters(
        [
            (f"{p}audio_tower.conv.weight", 300, False),
            (f"{p}multi_modal_projector.linear_1.weight", 50, False),
            (f"{p}{q}.weight", 600, False),
            (f"{p}{q}.lora_A.default.weight", 40, True),
            ("base_model.model.lm_head.weight", 10, False),
        ]
    )
    assert (count.total, count.trainable) == (1000, 40)
    assert count.trainable_pct == pytest.approx(4.0)
    assert count.by_component["language_model"] == {"total": 640, "trainable": 40}
    assert count.by_component["audio_tower"] == {"total": 300, "trainable": 0}
    assert count.by_component["other"] == {"total": 10, "trainable": 0}


class TestRelativeDelta:
    def test_zero_for_identical(self) -> None:
        a = np.ones((3, 3), dtype=np.float32)
        assert relative_delta(a, a.copy()) == 0.0

    def test_frobenius_ratio(self) -> None:
        base = np.full((2, 2), 2.0, dtype=np.float32)  # norm 4
        assert relative_delta(base, base + 1.0) == pytest.approx(0.5)  # ||ones|| = 2

    def test_zero_base_does_not_divide_by_zero(self) -> None:
        zero = np.zeros(4, dtype=np.float32)
        assert relative_delta(zero, zero + 1.0) == pytest.approx(2.0)


class TestSummarizeDeltas:
    def test_ok_when_only_targets_changed(self) -> None:
        report = summarize_deltas({Q: 0.01, DOWN: 0.02, ENC: 0.0, EMB: 0.0}, PATTERN)
        assert report.ok
        assert report.changed == {DOWN: 0.02, Q: 0.01}
        assert report.unchanged == 2

    def test_flags_changes_outside_the_decoder(self) -> None:
        report = summarize_deltas({Q: 0.01, ENC: 1e-6}, PATTERN)
        assert not report.ok
        assert report.unexpected_changes == [ENC]

    def test_not_ok_when_nothing_changed(self) -> None:
        report = summarize_deltas({Q: 0.0, ENC: 0.0}, PATTERN)
        assert not report.ok
        assert report.unchanged_targets == [Q]
