"""Tests for Trainer arguments, the training summary and the run guards (no torch needed)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from asr_assess.core.config import LoraTrainConfig, load_config
from asr_assess.core.manifest import write_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.training import trainer
from asr_assess.training.diagnostics import ParamCount
from asr_assess.training.trainer import TrainOutcome, training_kwargs, write_train_summary
from tests.fakes import write_clips

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def config(tmp_path: Path | None = None) -> LoraTrainConfig:
    cfg = load_config(CONFIGS / "lora.yaml", LoraTrainConfig)
    if tmp_path is None:
        return cfg
    return cfg.model_copy(update={
        "train_manifest": tmp_path / "train.jsonl", "dev_manifest": tmp_path / "dev.jsonl",
        "output_dir": tmp_path / "ckpt", "results_dir": tmp_path / "res",
    })  # fmt: skip


def record() -> RunRecord:
    return RunRecord(
        git_commit="abc",
        git_dirty=False,
        config={},
        config_hash="h",
        gpu_name=None,
        python_version="3.12",
        package_versions={},
        timestamp=datetime(2026, 9, 26, tzinfo=UTC),
    )


def outcome() -> TrainOutcome:
    return TrainOutcome(
        best_checkpoint="ckpt/checkpoint-42", best_metric=0.8, global_step=105,
        train_runtime_s=600.0, samples_per_second=2.7, peak_vram_gib=14.2, lora_modules=196,
        params=ParamCount(2000, 17, {"language_model": {"total": 1700, "trainable": 17}}),
        log_history=[{"loss": 1.2, "step": 5}, {"eval_loss": 0.8, "epoch": 1.0}],
        adapter_dir="ckpt/best",
    )  # fmt: skip


class TestTrainingKwargs:
    def test_maps_the_agreed_hyperparameters(self) -> None:
        kw = training_kwargs(config(), Path("out"), smoke=False)
        assert kw["learning_rate"] == 1e-4
        assert kw["lr_scheduler_type"] == "cosine"
        assert kw["warmup_steps"] == 0.1  # Transformers 5 reads a float < 1 as a ratio
        assert (kw["per_device_train_batch_size"], kw["gradient_accumulation_steps"]) == (8, 2)
        assert kw["num_train_epochs"] == 5
        assert kw["bf16"] and kw["gradient_checkpointing"]
        assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    def test_keeps_the_best_epoch_by_dev_loss(self) -> None:
        kw = training_kwargs(config(), Path("out"), smoke=False)
        assert kw["eval_strategy"] == kw["save_strategy"] == "epoch"
        assert kw["load_best_model_at_end"] is True
        assert kw["metric_for_best_model"] == "eval_loss"
        assert kw["greater_is_better"] is False
        assert kw["save_total_limit"] == 2
        assert kw["remove_unused_columns"] is False
        assert kw["report_to"] == "none"

    def test_smoke_runs_a_few_steps_and_evaluates_at_the_end(self) -> None:
        kw = training_kwargs(config(), Path("out"), smoke=True)
        assert kw["max_steps"] == 2
        assert kw["eval_strategy"] == kw["save_strategy"] == "steps"
        assert kw["eval_steps"] == kw["save_steps"] == 2


def test_summary_and_log_history_are_written(tmp_path: Path) -> None:
    write_train_summary(tmp_path / "run", record(), outcome(), {"train_clips": 322})
    summary = json.loads((tmp_path / "run" / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["record"]["git_commit"] == "abc"
    assert summary["params"]["trainable"] == 17
    assert summary["params"]["trainable_pct"] == pytest.approx(0.85)
    assert summary["lora_modules"] == 196
    assert summary["data"] == {"train_clips": 322}
    assert "log_history" not in summary
    lines = (tmp_path / "run" / "log_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == outcome().log_history


class TestRunTraining:
    def manifests(self, tmp_path: Path, dev: bool = True) -> None:
        entries = write_clips(tmp_path, [("a", 3.0, "2-5", "satu"), ("b", 6.0, "5-15", "dua")])
        write_manifest(tmp_path / "train.jsonl", entries)
        write_manifest(tmp_path / "dev.jsonl", entries[:1] if dev else [])

    def fake_train(self, calls: list[dict[str, Any]]) -> Any:
        def fake(cfg: Any, train_set: Any, dev_set: Any, out_dir: Path, smoke: bool) -> Any:
            calls.append({"train": len(train_set), "dev": len(dev_set), "out": out_dir})
            return outcome()

        return fake

    def test_runs_and_writes_the_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.manifests(tmp_path)
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(trainer, "train", self.fake_train(calls))
        trainer.run_training(config(tmp_path), "r1", smoke=False, record=record())
        assert calls == [{"train": 2, "dev": 1, "out": tmp_path / "ckpt" / "r1"}]
        summary = json.loads((tmp_path / "res" / "r1" / "train_summary.json").read_text("utf-8"))
        assert summary["data"]["train_clips"] == 2
        assert summary["data"]["train_minutes"] == pytest.approx(0.15)

    def test_smoke_takes_the_first_clips_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.manifests(tmp_path)
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(trainer, "train", self.fake_train(calls))
        cfg = config(tmp_path)
        cfg = cfg.model_copy(update={"smoke": cfg.smoke.model_copy(update={"train_clips": 1})})
        trainer.run_training(cfg, "s1", smoke=True, record=record())
        assert calls[0]["train"] == 1

    def test_refuses_existing_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.manifests(tmp_path)
        monkeypatch.setattr(trainer, "train", self.fake_train([]))
        (tmp_path / "res" / "r1").mkdir(parents=True)
        with pytest.raises(FileExistsError, match="never overwritten"):
            trainer.run_training(config(tmp_path), "r1", smoke=False, record=record())
        assert not (tmp_path / "ckpt" / "r1").exists()

    def test_empty_dev_manifest_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.manifests(tmp_path, dev=False)
        monkeypatch.setattr(trainer, "train", self.fake_train([]))
        with pytest.raises(ValueError, match="dev"):
            trainer.run_training(config(tmp_path), "r1", smoke=False, record=record())
