"""Tests for YAML config loading and the shipped config files."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from asr_assess.core.config import (
    BenchConfig,
    DataConfig,
    EngineConfig,
    EvalConfig,
    LoadTestConfig,
    LoraTrainConfig,
    StrictModel,
    load_config,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class Toy(StrictModel):
    name: str
    size: int


class TestLoadConfig:
    def test_loads_and_validates(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.yaml"
        path.write_text("name: a\nsize: 3\n", encoding="utf-8")
        assert load_config(path, Toy) == Toy(name="a", size=3)

    def test_rejects_unknown_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.yaml"
        path.write_text("name: a\nsize: 3\ntypo: 1\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(path, Toy)

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "toy.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(path, Toy)


class TestShippedConfigs:
    def test_lora_yaml_holds_the_agreed_defaults(self) -> None:
        cfg = load_config(CONFIGS / "lora.yaml", LoraTrainConfig)
        assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout) == (16, 32, 0.05)
        assert cfg.optim.learning_rate == 1e-4
        assert cfg.optim.per_device_batch_size * cfg.optim.gradient_accumulation_steps == 16
        assert cfg.optim.num_epochs == 5

    def test_loadtest_yaml_holds_the_spec_levels(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig)
        assert cfg.concurrency_levels == [1, 2, 4, 8, 16, 32, 64, 128]
        assert (cfg.warmup_s, cfg.steady_state_s, cfg.repeats) == (30.0, 120.0, 3)
        assert cfg.thresholds.p95_rtf_max == 0.5


class TestDataConfig:
    def load(self) -> dict[str, object]:
        return load_config(CONFIGS / "data.yaml", DataConfig).model_dump()

    def test_shipped_yaml_matches_the_minute_budget(self) -> None:
        cfg = load_config(CONFIGS / "data.yaml", DataConfig)
        assert sum(c.train.minutes for c in cfg.categories) == 35
        assert sum(c.eval.minutes for c in cfg.categories) == 12
        assert cfg.control.minutes == 5
        assert (cfg.min_duration_s, cfg.max_duration_s) == (2.0, 30.0)
        assert [b.name for b in cfg.buckets] == ["2-5", "5-15", "15-30"]

    def test_rejects_unknown_source_reference(self) -> None:
        raw = self.load()
        raw["control"]["source"] = "nope"  # type: ignore[index]
        with pytest.raises(ValidationError, match="nope"):
            DataConfig.model_validate(raw)

    def test_control_source_must_not_feed_a_category(self) -> None:
        raw = self.load()
        raw["control"]["source"] = raw["categories"][0]["eval"]["source"]  # type: ignore[index]
        with pytest.raises(ValidationError, match="control"):
            DataConfig.model_validate(raw)

    def test_eval_spread_must_be_able_to_fill_the_quota(self) -> None:
        raw = self.load()
        raw["eval_min_groups"], raw["eval_max_group_share"] = 2, 0.4
        with pytest.raises(ValidationError, match="eval_max_group_share"):
            DataConfig.model_validate(raw)

    def test_bucket_shares_must_sum_to_one(self) -> None:
        raw = self.load()
        raw["bucket_shares"] = {"2-5": 0.5, "5-15": 0.2, "15-30": 0.2}
        with pytest.raises(ValidationError, match="sum to 1"):
            DataConfig.model_validate(raw)

    def test_bucket_shares_must_name_every_bucket(self) -> None:
        raw = self.load()
        raw["bucket_shares"] = {"2-5": 0.5, "5-15": 0.5}
        with pytest.raises(ValidationError, match="bucket_shares"):
            DataConfig.model_validate(raw)

    def test_rejects_unsupported_language(self) -> None:
        raw = self.load()
        raw["categories"][0]["language"] = "Klingon"  # type: ignore[index]
        with pytest.raises(ValidationError, match="Klingon"):
            DataConfig.model_validate(raw)


class TestLoadTestValidation:
    def test_levels_must_be_strictly_ascending(self, tmp_path: Path) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig).model_dump()
        cfg["concurrency_levels"] = [1, 4, 2]
        with pytest.raises(ValidationError, match="ascending"):
            LoadTestConfig.model_validate(cfg)

    def test_strong_threshold_must_not_exceed_max(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig).model_dump()
        cfg["thresholds"]["p95_rtf_strong"] = 0.6
        with pytest.raises(ValidationError, match="p95_rtf_strong"):
            LoadTestConfig.model_validate(cfg)


class TestInferenceConfigs:
    def test_baseline_engine_is_plain_transformers(self) -> None:
        cfg = load_config(CONFIGS / "engines" / "hf_base.yaml", EngineConfig)
        assert (cfg.kind, cfg.model_id) == ("hf", "Qwen/Qwen3-ASR-1.7B-hf")
        assert cfg.attn_implementation == "eager"
        assert cfg.language_hint == "auto"

    def test_eval_config_is_batch_one_over_eval_and_control(self) -> None:
        cfg = load_config(CONFIGS / "eval.yaml", EvalConfig)
        assert cfg.batch_size == 1
        assert set(cfg.manifests) == {"eval", "control"}
        assert 0 < cfg.bootstrap.confidence < 1

    def test_bench_config_loads(self) -> None:
        cfg = load_config(CONFIGS / "bench.yaml", BenchConfig)
        assert cfg.clips_per_bucket > 0 and cfg.repeats > 0

    def test_confidence_must_be_a_probability(self) -> None:
        raw = load_config(CONFIGS / "eval.yaml", EvalConfig).model_dump()
        raw["bootstrap"]["confidence"] = 1.5
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(raw)
