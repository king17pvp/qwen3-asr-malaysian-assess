"""Plain Transformers backend: the baseline engine.

API as documented for Transformers' ``qwen3_asr``: ``apply_transcription_request`` builds the
chat-formatted inputs, ``generate`` decodes greedily, ``decode(..., return_format="parsed")``
splits ``language X<asr_text>...`` into language and transcription.
"""

import logging
from collections.abc import Sequence
from typing import Any

from asr_assess.core.config import EngineConfig
from asr_assess.inference.engine import AudioRequest, Transcript

log = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024**3


def language_argument(languages: Sequence[str | None]) -> str | list[str | None] | None:
    """The ``language`` argument for a batch: None (auto), one shared name, or one per clip."""
    unique = set(languages)
    if unique == {None}:
        return None
    if len(unique) == 1:
        return languages[0]
    return list(languages)


def hit_token_limit(generated_tokens: int, max_new_tokens: int) -> bool:
    """Whether generation stopped because it ran out of tokens, not at end-of-sequence."""
    return generated_tokens >= max_new_tokens


class HFEngine:
    """Qwen3-ASR through ``Qwen3ASRForConditionalGeneration`` with greedy decoding."""

    def __init__(self, cfg: EngineConfig) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.name = f"hf:{cfg.model_id}:{cfg.attn_implementation}"
        self._cfg = cfg
        self._torch: Any = torch
        self._processor: Any = AutoProcessor.from_pretrained(cfg.model_id)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            cfg.model_id,
            dtype=getattr(torch, cfg.dtype),
            attn_implementation=cfg.attn_implementation,
        )
        self._model: Any = model.to(cfg.device).eval()
        self._pad_id = self._processor.tokenizer.pad_token_id
        log.info("Loaded %s on %s", self.name, cfg.device)

    def transcribe(self, batch: Sequence[AudioRequest]) -> list[Transcript]:
        """Greedy transcription of ``batch`` in one ``generate`` call."""
        inputs = self._processor.apply_transcription_request(
            audio=[r.samples for r in batch],
            language=language_argument([r.language for r in batch]),
        ).to(self._model.device, self._model.dtype)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self._cfg.max_new_tokens, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        parsed = self._processor.decode(generated, return_format="parsed")
        lengths = (generated != self._pad_id).sum(dim=1).tolist()
        return [
            Transcript(
                id=request.id,
                text=item["transcription"].strip(),
                language=item["language"] or None,
                hit_token_limit=hit_token_limit(length, self._cfg.max_new_tokens),
            )
            for request, item, length in zip(batch, parsed, lengths, strict=True)
        ]

    def peak_memory_gb(self) -> float:
        """Peak CUDA memory allocated by this process since the last reset."""
        if not self._torch.cuda.is_available():
            return 0.0
        peak: int = self._torch.cuda.max_memory_allocated()
        return peak / _BYTES_PER_GIB

    def reset_peak_memory(self) -> None:
        """Start a new peak-memory window."""
        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
