"""Tests for YAML config loading and the shipped config files."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from asr_assess.core.config import (
    BenchConfig,
    DataConfig,
    EvalConfig,
    HFEngineConfig,
    LoadProfile,
    LoadTestConfig,
    LoraTrainConfig,
    ServeConfig,
    StrictModel,
    VLLMHTTPEngineConfig,
    VLLMServeConfig,
    load_config,
    load_engine_config,
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
        # Checkpoint selection uses dev; the reported eval set is never seen during training.
        assert cfg.dev_manifest == Path("data/manifests/dev.jsonl")

    def test_lora_yaml_holds_the_training_paths(self) -> None:
        cfg = load_config(CONFIGS / "lora.yaml", LoraTrainConfig)
        assert cfg.merged_dir == Path("checkpoints/merged")
        assert cfg.results_dir == Path("results/train")
        assert cfg.merge_results_dir == Path("results/merge")
        assert cfg.attn_implementation == "sdpa"
        assert (cfg.smoke.train_clips, cfg.smoke.dev_clips, cfg.smoke.max_steps) == (8, 4, 2)
        assert cfg.optim.save_total_limit == 2

    def test_fine_tuned_engine_differs_from_baseline_only_in_weights(self) -> None:
        base = load_config(CONFIGS / "engines" / "hf_base.yaml", HFEngineConfig)
        tuned = load_config(CONFIGS / "engines" / "hf_ft.yaml", HFEngineConfig)
        assert tuned.model_id == "king17pvp/qwen3-asr-1.7b-malaysian"
        assert tuned.model_copy(update={"model_id": base.model_id}) == base

    def test_loadtest_yaml_holds_the_spec_levels(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig)
        full, quick = cfg.profiles["full"], cfg.profiles["quick"]
        assert full.concurrency_levels == [1, 2, 4, 8, 16, 32, 64, 128]
        assert (full.warmup_s, full.steady_state_s, full.repeats) == (30.0, 120.0, 3)
        assert quick.repeats == 1
        assert cfg.thresholds.p95_rtf_max == 0.5

    def test_loadtest_yaml_caps_client_lag(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig)
        assert cfg.thresholds.max_client_lag_s == 0.5

    def test_loadtest_yaml_has_the_open_live_profile(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig)
        live = cfg.profiles["open_live"]
        assert (live.mode, live.pause_s) == ("open", 1.0)
        assert live.concurrency_levels == [1, 64, 128, 256, 384, 512, 768]
        assert (live.warmup_s, live.steady_state_s, live.repeats) == (30.0, 120.0, 1)
        assert (live.bisect, live.bisect_max_steps) == (True, 5)
        assert all(cfg.profiles[name].mode == "closed" for name in ("full", "quick"))


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

    def test_shipped_yaml_has_the_dev_budget(self) -> None:
        cfg = load_config(CONFIGS / "data.yaml", DataConfig)
        dev = {c.name: c.dev_minutes for c in cfg.categories}
        assert dev == {
            "manglish": 2.0,
            "malay_conversational": 0.5,
            "malay_read": 0.5,
            "english_read": 0.5,
        }

    def test_dev_minutes_is_optional(self) -> None:
        raw = self.load()
        for category in raw["categories"]:  # type: ignore[attr-defined]
            category.pop("dev_minutes", None)
        cfg = DataConfig.model_validate(raw)
        assert all(c.dev_minutes is None for c in cfg.categories)

    def test_dev_minutes_must_be_positive(self) -> None:
        raw = self.load()
        raw["categories"][0]["dev_minutes"] = 0  # type: ignore[index]
        with pytest.raises(ValidationError, match="dev_minutes"):
            DataConfig.model_validate(raw)


class TestLoadTestValidation:
    def profile(self, levels: list[int]) -> dict[str, object]:
        return {
            "concurrency_levels": levels,
            "repeats": 1,
            "warmup_s": 1,
            "steady_state_s": 2,
            "bisect": True,
            "bisect_max_steps": 3,
            "stop_after_failures": 1,
        }

    def test_levels_must_be_strictly_ascending(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            LoadProfile.model_validate(self.profile([1, 4, 2]))

    def test_levels_must_start_at_one(self) -> None:
        with pytest.raises(ValidationError, match="start at 1"):
            LoadProfile.model_validate(self.profile([2, 4]))

    def test_profiles_default_to_closed_loop(self) -> None:
        profile = LoadProfile.model_validate(self.profile([1, 2]))
        assert (profile.mode, profile.pause_s) == ("closed", 0.0)

    def test_pause_needs_open_loop(self) -> None:
        with pytest.raises(ValidationError, match="pause_s must be 0"):
            LoadProfile.model_validate({**self.profile([1, 2]), "pause_s": 1.0})

    def test_pause_must_not_be_negative(self) -> None:
        with pytest.raises(ValidationError, match="pause_s"):
            LoadProfile.model_validate({**self.profile([1, 2]), "mode": "open", "pause_s": -1.0})

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mode"):
            LoadProfile.model_validate({**self.profile([1, 2]), "mode": "half-open"})

    def test_strong_threshold_must_not_exceed_max(self) -> None:
        cfg = load_config(CONFIGS / "loadtest.yaml", LoadTestConfig).model_dump()
        cfg["thresholds"]["p95_rtf_strong"] = 0.6
        with pytest.raises(ValidationError, match="p95_rtf_strong"):
            LoadTestConfig.model_validate(cfg)


class TestEngineConfigs:
    def test_shipped_hf_engines_load_as_hf(self) -> None:
        for name in ("hf_base.yaml", "hf_ft.yaml"):
            assert isinstance(load_engine_config(CONFIGS / "engines" / name), HFEngineConfig)

    def test_vllm_http_engine(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text(
            "kind: vllm_http\nbase_url: http://localhost:8000\n"
            "request_timeout_s: 60\nmax_new_tokens: 256\nlanguage_hint: auto\n"
        )
        cfg = load_engine_config(path)
        assert isinstance(cfg, VLLMHTTPEngineConfig)
        assert cfg.model is None

    def test_vllm_http_rejects_manifest_language(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text(
            "kind: vllm_http\nbase_url: http://x\nrequest_timeout_s: 1\n"
            "max_new_tokens: 8\nlanguage_hint: manifest\n"
        )
        with pytest.raises(ValidationError):
            load_engine_config(path)

    def test_unknown_kind_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text("kind: onnx\n")
        with pytest.raises(ValidationError):
            load_engine_config(path)


class TestServeConfigs:
    def test_serve_config_validates(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text(
            "engine: configs/engines/hf_ft.yaml\nhost: 0.0.0.0\nport: 8001\n"
            "max_batch: 1\nmax_wait_ms: 0\nsample_rate: 16000\n"
        )
        assert load_config(path, ServeConfig).max_batch == 1

    def test_vllm_serve_config_defaults_to_stock(self, tmp_path: Path) -> None:
        path = tmp_path / "v.yaml"
        path.write_text("model: checkpoints/merged/lora\nport: 8000\n")
        cfg = load_config(path, VLLMServeConfig)
        assert cfg.enforce_eager is False
        assert cfg.max_num_seqs is None
        assert cfg.extra_args == []

    def test_gpu_memory_utilization_is_a_fraction(self, tmp_path: Path) -> None:
        path = tmp_path / "v.yaml"
        path.write_text("model: m\nport: 8000\ngpu_memory_utilization: 1.5\n")
        with pytest.raises(ValidationError):
            load_config(path, VLLMServeConfig)


class TestInferenceConfigs:
    def test_baseline_engine_is_plain_transformers(self) -> None:
        cfg = load_config(CONFIGS / "engines" / "hf_base.yaml", HFEngineConfig)
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
