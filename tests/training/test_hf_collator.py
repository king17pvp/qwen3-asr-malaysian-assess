"""The collator against the real Qwen3-ASR processor (tokenizer files only, no weights).

Run with: uv run --extra train pytest -m hf
"""

from typing import Any

import numpy as np
import pytest

from asr_assess.training.collator import IGNORE_INDEX, ASRCollator, Example

pytestmark = pytest.mark.hf
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
RATE = 16000
TARGET = "language None<asr_text>boleh lah, see you esok"


@pytest.fixture(scope="module")
def processor() -> Any:
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def collator(processor: Any) -> ASRCollator:
    torch = pytest.importorskip("torch")
    return ASRCollator(processor, torch.bfloat16)


def audio(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.01 * rng.standard_normal(int(seconds * RATE))).astype(np.float32)


def reply(processor: Any, batch: dict[str, Any], row: int) -> str:
    labels = [t for t in batch["labels"][row].tolist() if t != IGNORE_INDEX]
    return str(processor.tokenizer.decode(labels))


def test_training_prompt_matches_inference_prompt(processor: Any, collator: ASRCollator) -> None:
    samples = audio(3.0)
    inference = processor.apply_transcription_request(audio=[samples])["input_ids"][0].tolist()
    training = collator([Example(samples, TARGET)])["input_ids"][0].tolist()
    assert training[: len(inference)] == inference


def test_labels_are_exactly_the_reply(processor: Any, collator: ASRCollator) -> None:
    batch = collator([Example(audio(3.0), TARGET)])
    assert reply(processor, batch, 0).startswith(TARGET + "<|im_end|>")
    assert reply(processor, batch, 0).rstrip("\n") == TARGET + "<|im_end|>"


def test_mixed_lengths_give_the_same_labels(processor: Any, collator: ASRCollator) -> None:
    alone = collator([Example(audio(2.0), TARGET)])
    mixed = collator(
        [Example(audio(2.0), TARGET), Example(audio(20.0), "language English<asr_text>hi")]
    )
    assert reply(processor, mixed, 0) == reply(processor, alone, 0)
    assert reply(processor, mixed, 1).startswith("language English<asr_text>hi<|im_end|>")


def test_batch_holds_only_model_inputs_in_the_training_dtype(collator: ASRCollator) -> None:
    torch = pytest.importorskip("torch")
    batch = collator([Example(audio(2.0), TARGET)])
    assert set(batch) == {
        "input_ids", "attention_mask", "input_features", "input_features_mask", "labels"
    }  # fmt: skip
    assert batch["input_features"].dtype == torch.bfloat16
    assert batch["labels"].shape == batch["input_ids"].shape
