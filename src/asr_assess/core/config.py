"""Typed configs for every YAML file, and a single loader.

Data, eval, serve and vLLM models are added alongside the stages that use them.
"""

from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    """Immutable config model that rejects unknown keys, so typos fail loudly."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def load_config[T: StrictModel](path: Path, model: type[T]) -> T:
    """Load a YAML file and validate it against ``model``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return model.model_validate(raw)


# ---------------------------------------------------------------- training


class LoraSettings(StrictModel):
    """LoRA adapter shape; target modules are matched in the decoder only."""

    rank: PositiveInt
    alpha: PositiveInt
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)


class OptimSettings(StrictModel):
    """Optimizer, schedule and batching; names follow HF TrainingArguments where they map 1:1."""

    learning_rate: PositiveFloat
    lr_scheduler_type: Literal["cosine", "linear", "constant_with_warmup"]
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    per_device_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    num_epochs: PositiveInt
    bf16: bool
    gradient_checkpointing: bool
    eval_strategy: Literal["epoch", "steps"]
    metric_for_best_model: str
    logging_steps: PositiveInt


class LoraTrainConfig(StrictModel):
    """configs/lora.yaml: decoder-only LoRA fine-tuning of Qwen3-ASR."""

    model_id: str
    train_manifest: Path
    eval_manifest: Path
    output_dir: Path
    seed: int
    lora: LoraSettings
    optim: OptimSettings


# ---------------------------------------------------------------- load test


class LoadTestThresholds(StrictModel):
    """Pass criteria for "max sustainable" concurrency."""

    p95_rtf_max: PositiveFloat
    p95_rtf_strong: PositiveFloat
    max_wer_delta_points: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _strong_within_max(self) -> Self:
        if self.p95_rtf_strong > self.p95_rtf_max:
            raise ValueError("p95_rtf_strong must not exceed p95_rtf_max")
        return self


class AudioPoolSettings(StrictModel):
    """The seeded utterance pool replayed identically in every run."""

    manifest: Path
    size: PositiveInt
    seed: int


class LoadTestConfig(StrictModel):
    """configs/loadtest.yaml: closed-loop N-stream load test."""

    concurrency_levels: list[PositiveInt] = Field(min_length=1)
    bisect_near_limit: bool
    bisect_max_steps: PositiveInt
    warmup_s: float = Field(ge=0.0)
    steady_state_s: PositiveFloat
    repeats: PositiveInt
    request_timeout_s: PositiveFloat
    gpu_sample_hz: PositiveFloat
    audio_pool: AudioPoolSettings
    thresholds: LoadTestThresholds
    output_dir: Path

    @model_validator(mode="after")
    def _levels_ascending(self) -> Self:
        levels = self.concurrency_levels
        if any(b <= a for a, b in pairwise(levels)):
            raise ValueError("concurrency_levels must be strictly ascending")
        return self
