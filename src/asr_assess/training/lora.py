"""LoRA adapters on the decoder LLM only; the audio encoder and projector stay frozen."""

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from asr_assess.core.config import LoraSettings

log = logging.getLogger(__name__)
_DECODER = "model.language_model."
_ADAPTER_CHILD = ".lora_A"


def decoder_target_pattern(names: Sequence[str]) -> str:
    """Regex (PEFT full-matches strings) selecting ``names`` under the decoder LLM only.

    A plain name list would also match the audio encoder's own q_proj/k_proj/v_proj.
    """
    alternatives = "|".join(re.escape(name) for name in names)
    return rf"{re.escape(_DECODER)}.*\.({alternatives})"


def lora_module_names(module_names: Iterable[str]) -> list[str]:
    """Modules that received an adapter: those owning a ``lora_A`` child."""
    return sorted({n.split(_ADAPTER_CHILD)[0] for n in module_names if _ADAPTER_CHILD in n})


def check_decoder_only(wrapped: Sequence[str]) -> None:
    """Raise unless at least one module is wrapped and every wrapped module is in the decoder."""
    if not wrapped:
        raise RuntimeError("LoRA wrapped no modules; check lora.target_modules")
    outside = [name for name in wrapped if f".{_DECODER}" not in f".{name}"]
    if outside:
        raise RuntimeError(f"LoRA wrapped modules outside the decoder: {outside[:3]}")


def apply_lora(model: Any, settings: LoraSettings) -> tuple[Any, int]:
    """Wrap the decoder projections with LoRA; returns the PEFT model and the module count."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=decoder_target_pattern(settings.target_modules),
    )
    peft_model = get_peft_model(model, config)
    wrapped = lora_module_names(name for name, _ in peft_model.named_modules())
    check_decoder_only(wrapped)
    log.info("LoRA on %d decoder modules", len(wrapped))
    return peft_model, len(wrapped)
