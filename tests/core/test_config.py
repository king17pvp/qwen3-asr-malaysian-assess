"""Tests for YAML config loading and the shipped config files."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from asr_assess.core.config import LoadTestConfig, LoraTrainConfig, StrictModel, load_config

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
