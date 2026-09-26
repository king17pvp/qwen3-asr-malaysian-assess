"""Tests for the post-merge smoke transcription and the merge run guards."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from asr_assess.core.config import EngineConfig, LoraTrainConfig, load_config
from asr_assess.core.manifest import write_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.training import export
from asr_assess.training.export import smoke_transcribe
from tests.fakes import ScriptedEngine, write_clips

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


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


class TestSmokeTranscribe:
    def test_rows_pair_reference_and_hypothesis(self, tmp_path: Path) -> None:
        entries = write_clips(tmp_path, [("a", 3.0, "2-5", "satu dua")])
        rows = smoke_transcribe(ScriptedEngine({entries[0].audio: "satu dua"}), entries, 16000)
        # Auto language detection (as served): the language is whatever the engine reports.
        assert rows == [{"id": entries[0].audio, "reference": "satu dua",
                         "hypothesis": "satu dua", "language": None}]  # fmt: skip

    def test_empty_hypothesis_fails(self, tmp_path: Path) -> None:
        entries = write_clips(tmp_path, [("a", 3.0, "2-5", "satu")])
        with pytest.raises(RuntimeError, match="empty"):
            smoke_transcribe(ScriptedEngine({}), entries, 16000)


class TestRunMerge:
    def setup(self, tmp_path: Path) -> tuple[LoraTrainConfig, EngineConfig]:
        entries = write_clips(tmp_path, [("a", 3.0, "2-5", "satu")])
        write_manifest(tmp_path / "dev.jsonl", entries)
        cfg = load_config(CONFIGS / "lora.yaml", LoraTrainConfig).model_copy(update={
            "dev_manifest": tmp_path / "dev.jsonl", "output_dir": tmp_path / "ckpt",
            "merged_dir": tmp_path / "merged", "merge_results_dir": tmp_path / "res",
        })  # fmt: skip
        return cfg, load_config(CONFIGS / "engines" / "hf_ft.yaml", EngineConfig)

    def test_missing_adapter_is_actionable(self, tmp_path: Path) -> None:
        cfg, engine_cfg = self.setup(tmp_path)
        with pytest.raises(FileNotFoundError, match="asr-assess train"):
            export.run_merge(cfg, engine_cfg, "r1", record())

    def test_refuses_existing_merge(self, tmp_path: Path) -> None:
        cfg, engine_cfg = self.setup(tmp_path)
        (tmp_path / "ckpt" / "r1" / "best").mkdir(parents=True)
        (tmp_path / "merged" / "r1").mkdir(parents=True)
        with pytest.raises(FileExistsError, match="never overwritten"):
            export.run_merge(cfg, engine_cfg, "r1", record())

    def test_writes_deltas_smoke_and_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, engine_cfg = self.setup(tmp_path)
        (tmp_path / "ckpt" / "r1" / "best").mkdir(parents=True)
        loaded: list[str] = []
        q = "model.language_model.layers.0.self_attn.q_proj.weight"
        merged = tmp_path / "merged" / "r1"
        monkeypatch.setattr(export, "merge_adapter", lambda *a: merged.mkdir(parents=True))
        monkeypatch.setattr(export, "base_snapshot", lambda model_id: tmp_path / "base")
        monkeypatch.setattr(export, "weight_deltas", lambda base, tuned: {q: 0.01})

        def fake_engine(ec: EngineConfig) -> Any:
            loaded.append(ec.model_id)
            return ScriptedEngine({str(tmp_path / "a.wav"): "satu"})

        monkeypatch.setattr("asr_assess.inference.factory.make_engine", fake_engine)
        report = export.run_merge(cfg, engine_cfg, "r1", record())
        assert report.ok
        assert loaded == [str(tmp_path / "merged" / "r1")]
        out = tmp_path / "res" / "r1"
        assert json.loads((out / "weight_deltas.json").read_text("utf-8"))["changed"] == {q: 0.01}
        assert (out / "smoke.jsonl").read_text("utf-8").count("\n") == 1
        summary = json.loads((out / "merge_summary.json").read_text("utf-8"))
        assert summary["record"]["git_commit"] == "abc"

    def test_unexpected_change_fails_after_writing_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, engine_cfg = self.setup(tmp_path)
        (tmp_path / "ckpt" / "r1" / "best").mkdir(parents=True)
        enc = "model.audio_tower.layers.0.fc1.weight"
        monkeypatch.setattr(export, "merge_adapter", lambda *a: None)
        monkeypatch.setattr(export, "base_snapshot", lambda model_id: tmp_path / "base")
        monkeypatch.setattr(export, "weight_deltas", lambda base, tuned: {enc: 0.5})
        with pytest.raises(RuntimeError, match="outside"):
            export.run_merge(cfg, engine_cfg, "r1", record())
        assert (tmp_path / "res" / "r1" / "weight_deltas.json").exists()
