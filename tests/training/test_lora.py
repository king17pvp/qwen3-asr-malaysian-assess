"""Tests for decoder-only LoRA targeting."""

import re

import pytest

from asr_assess.training.lora import (
    check_decoder_only,
    decoder_target_pattern,
    lora_module_names,
)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Real module names from Qwen3ASRForConditionalGeneration (Transformers 5.17).
DECODER = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.27.mlp.gate_proj",
    "model.language_model.layers.27.mlp.up_proj",
    "model.language_model.layers.27.mlp.down_proj",
]
NOT_DECODER = [
    "model.audio_tower.layers.0.self_attn.q_proj",
    "model.audio_tower.layers.0.self_attn.k_proj",
    "model.audio_tower.layers.0.self_attn.v_proj",
    "model.audio_tower.layers.0.self_attn.out_proj",
    "model.audio_tower.layers.0.fc1",
    "model.multi_modal_projector.linear_1",
    "model.language_model.embed_tokens",
    "model.language_model.layers.0.self_attn.q_norm",
    "lm_head",
]


class TestDecoderTargetPattern:
    @pytest.mark.parametrize("name", DECODER)
    def test_matches_decoder_projections(self, name: str) -> None:
        assert re.fullmatch(decoder_target_pattern(TARGETS), name)

    @pytest.mark.parametrize("name", NOT_DECODER)
    def test_never_matches_encoder_projector_embeddings_or_head(self, name: str) -> None:
        assert not re.fullmatch(decoder_target_pattern(TARGETS), name)

    def test_escapes_names(self) -> None:
        assert not re.fullmatch(decoder_target_pattern(["q.proj"]), "model.language_model.x.qXproj")


def test_lora_module_names_collapses_adapter_children() -> None:
    names = [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default",
        "base_model.model.model.audio_tower.layers.0.fc1",
    ]
    assert lora_module_names(names) == [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    ]


class TestCheckDecoderOnly:
    def test_accepts_decoder_modules(self) -> None:
        check_decoder_only(["base_model.model.model.language_model.layers.0.self_attn.q_proj"])

    def test_rejects_an_encoder_module(self) -> None:
        with pytest.raises(RuntimeError, match="audio_tower"):
            check_decoder_only(["base_model.model.model.audio_tower.layers.0.self_attn.q_proj"])

    def test_rejects_nothing_wrapped(self) -> None:
        with pytest.raises(RuntimeError, match="no modules"):
            check_decoder_only([])
