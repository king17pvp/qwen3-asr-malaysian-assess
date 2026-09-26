"""Typed configs for every YAML file, and a single loader.

Eval, serve and vLLM models are added alongside the stages that use them.
"""

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from asr_assess.core.audio import Bucket
from asr_assess.core.text import SUPPORTED_LANGUAGES


class StrictModel(BaseModel):
    """Immutable config model that rejects unknown keys, so typos fail loudly."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def load_config[T: StrictModel](path: Path, model: type[T]) -> T:
    """Load a YAML file and validate it against ``model``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return model.model_validate(raw)


# ---------------------------------------------------------------- data


class BucketSpec(StrictModel):
    """A named duration bucket in seconds."""

    name: str
    min_s: float = Field(ge=0.0)
    max_s: PositiveFloat

    def to_bucket(self) -> Bucket:
        """The runtime bucket used by ``core.audio.assign_bucket``."""
        return Bucket(self.name, self.min_s, self.max_s)


class MesoliticaContextSource(StrictModel):
    """Malaysian YouTube clips: text from one parquet row group, audio from remote zips."""

    kind: Literal["mesolitica_context"]
    repo: str
    parquet: str
    row_group: int = Field(ge=0)
    zips: list[str] = Field(min_length=1)
    # prepared-pseudolabel.jsonl maps a clip's row index to its YouTube video; only the first
    # ``index_bytes`` are read, which bounds the candidate pool.
    index_repo: str
    index_path: str
    index_bytes: PositiveInt
    # Videos whose transcript is more than this share Indonesian-only words are excluded.
    indonesian_max_share: float = Field(ge=0.0, le=1.0)
    indonesian_min_hits: PositiveInt


class MesoliticaConversationalSource(StrictModel):
    """Malay Conversational Speech Corpus clips, joined to the original corpus for speaker ids."""

    kind: Literal["mesolitica_conversational"]
    repo: str
    parquet: str
    zip: str
    prefix: str
    speakers_repo: str
    speakers_parquet: str


class FleursSource(StrictModel):
    """One split of FLEURS from its auto-converted parquet revision."""

    kind: Literal["fleurs"]
    repo: str
    revision: str
    config: str
    split: str


class LibriSpeechSource(StrictModel):
    """Selected row groups of a LibriSpeech parquet file (audio embedded)."""

    kind: Literal["librispeech"]
    repo: str
    parquet: str
    row_groups: list[int] = Field(min_length=1)


SourceSpec = Annotated[
    MesoliticaContextSource | MesoliticaConversationalSource | FleursSource | LibriSpeechSource,
    Field(discriminator="kind"),
]


class Quota(StrictModel):
    """Minutes of audio to take from a named source."""

    source: str
    minutes: PositiveFloat


class CategorySpec(StrictModel):
    """A language category with its train, held-out eval and optional dev quotas.

    When train and eval name the same source, whole speakers/videos go to one side only.
    Dev draws from the train source and is used only to pick the best fine-tuning checkpoint.
    """

    name: str
    language: str | None
    train: Quota
    eval: Quota
    dev_minutes: PositiveFloat | None = None
    eval_pool_factor: float | None = Field(default=None, ge=1.0)  # overrides the global one


class ControlSpec(Quota):
    """Out-of-domain control set, never trained on."""

    language: str


class DataConfig(StrictModel):
    """configs/data.yaml: dataset sampling, cleaning and splitting."""

    seed: int
    sample_rate: PositiveInt
    min_duration_s: PositiveFloat
    max_duration_s: PositiveFloat
    min_words: PositiveInt
    buckets: list[BucketSpec] = Field(min_length=1)
    bucket_shares: dict[str, float]
    eval_pool_factor: float = Field(ge=1.0)
    eval_min_groups: PositiveInt
    eval_max_group_share: float = Field(gt=0.0, le=1.0)
    output_dir: Path
    sources: dict[str, SourceSpec]
    categories: list[CategorySpec] = Field(min_length=1)
    control: ControlSpec

    def sources_in_use(self) -> list[str]:
        """Names of the sources some category or the control set draws from."""
        quotas = [q for c in self.categories for q in (c.train, c.eval)] + [self.control]
        return sorted({q.source for q in quotas})

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [b.name for b in self.buckets]
        if set(self.bucket_shares) != set(names):
            raise ValueError(f"bucket_shares keys must be exactly the bucket names {names}")
        if abs(sum(self.bucket_shares.values()) - 1.0) > 1e-6:
            raise ValueError("bucket_shares must sum to 1")
        if self.eval_min_groups * self.eval_max_group_share < 1.0:
            raise ValueError("eval_min_groups * eval_max_group_share must be >= 1 to fill eval")
        quotas = [q for c in self.categories for q in (c.train, c.eval)] + [self.control]
        for quota in quotas:
            if quota.source not in self.sources:
                raise ValueError(f"Unknown source {quota.source!r}")
        if self.control.source in {q.source for q in quotas[:-1]}:
            raise ValueError("The control source must not also feed a train/eval category")
        languages = [c.language for c in self.categories] + [self.control.language]
        for language in languages:
            if language is not None and language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported Qwen3-ASR language: {language!r}")
        return self


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


# ---------------------------------------------------------------- inference

# auto: the model detects the language; manifest: force each clip's manifest language.
LanguageHint = Literal["auto", "manifest"]


class EngineConfig(StrictModel):
    """configs/engines/*.yaml: which ASR backend to run and how to load it."""

    kind: Literal["hf"]
    model_id: str
    dtype: Literal["bfloat16", "float16", "float32"]
    device: str
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"]
    max_new_tokens: PositiveInt
    language_hint: LanguageHint


class BootstrapSettings(StrictModel):
    """Percentile bootstrap over utterances for WER/CER confidence intervals."""

    n_resamples: PositiveInt
    confidence: float = Field(gt=0.0, lt=1.0)
    seed: int


class EvalConfig(StrictModel):
    """configs/eval.yaml: WER/CER evaluation of an engine over named manifests."""

    manifests: dict[str, Path] = Field(min_length=1)
    batch_size: PositiveInt
    sample_rate: PositiveInt
    bootstrap: BootstrapSettings
    output_dir: Path


class BenchConfig(StrictModel):
    """configs/bench.yaml: single-stream RTF per duration bucket."""

    manifest: Path
    clips_per_bucket: PositiveInt
    warmup_requests: int = Field(ge=0)
    repeats: PositiveInt
    seed: int
    sample_rate: PositiveInt
    output_dir: Path
