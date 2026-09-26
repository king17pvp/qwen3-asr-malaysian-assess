"""Training examples, chat-formatted batches and label masking (loss on the reply only)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_assess.core.audio import Samples, load_audio
from asr_assess.core.manifest import ManifestEntry

IGNORE_INDEX = -100
# The assistant turn header in the Qwen3-ASR chat template; everything up to it is prompt.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class Example:
    """One training utterance: 16 kHz mono samples and its ``language X<asr_text>...`` target."""

    samples: Samples
    target: str


def load_examples(entries: Sequence[ManifestEntry], sample_rate: int) -> list[Example]:
    """Load every clip into memory (~35 min of audio is ~130 MB as float32)."""
    return [Example(load_audio(Path(e.audio), sample_rate), e.text) for e in entries]


def build_conversation(target: str) -> list[dict[str, Any]]:
    """Chat messages for one example: the same audio user turn inference sends, plus the reply.

    Inference (``apply_transcription_request`` without a prompt) sends no system message; the
    template still renders an empty system block, identically in both cases.
    """
    return [
        {"role": "user", "content": [{"type": "audio"}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


def mask_labels(
    input_ids: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    header: Sequence[int],
) -> list[list[int]]:
    """Causal-LM labels that score only the assistant reply.

    Positions up to and including the last ``header`` (system, user, audio and the header
    itself) and every position with attention mask 0 become ``IGNORE_INDEX``. Padding is found
    through the attention mask, never the pad id: for Qwen3-ASR the model's pad id is the EOS.
    """
    labels = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        start = _last_occurrence(ids, header)
        if start is None:
            raise ValueError("A batch row has no assistant header; check the chat template")
        reply_from = start + len(header)
        labels.append(
            [
                token if i >= reply_from and keep else IGNORE_INDEX
                for i, (token, keep) in enumerate(zip(ids, mask, strict=True))
            ]
        )
    return labels


def _last_occurrence(ids: Sequence[int], pattern: Sequence[int]) -> int | None:
    width, target = len(pattern), list(pattern)
    for start in range(len(ids) - width, -1, -1):
        if list(ids[start : start + width]) == target:
            return start
    return None
