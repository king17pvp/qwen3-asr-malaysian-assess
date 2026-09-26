"""Train -> merge -> reload on a tiny randomly initialised Qwen3-ASR, on CPU.

Uses the real processor and model classes with shrunk dimensions, so the PEFT, Trainer,
collator and export wiring runs for real in about a minute. Run with:
    uv run --extra train pytest -m hf tests/training/test_hf_end_to_end.py
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from asr_assess.core.config import EngineConfig, LoraTrainConfig, load_config
from asr_assess.core.manifest import write_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.inference import factory
from asr_assess.training import export
from asr_assess.training.trainer import run_training
from tests.fakes import ScriptedEngine, write_clips

pytestmark = pytest.mark.hf
CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


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


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    transformers = pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("tiny")
    config = transformers.Qwen3ASRConfig(
        audio_config={"d_model": 32, "encoder_layers": 1, "encoder_attention_heads": 2,
                      "encoder_ffn_dim": 64, "output_dim": 32, "downsample_hidden_size": 16},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                     "vocab_size": 151936, "tie_word_embeddings": True},
    )  # fmt: skip
    transformers.Qwen3ASRForConditionalGeneration(config).save_pretrained(str(out))
    transformers.AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(str(out))
    return out


def test_train_merge_reload(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = write_clips(tmp_path, [(f"c{i}", 2.0 + i, "2-5", f"satu {i}") for i in range(4)])
    write_manifest(tmp_path / "train.jsonl", entries)
    write_manifest(tmp_path / "dev.jsonl", entries[:2])
    base = load_config(CONFIGS / "lora.yaml", LoraTrainConfig)
    cfg = base.model_copy(update={
        "model_id": str(tiny_model_dir), "train_manifest": tmp_path / "train.jsonl",
        "dev_manifest": tmp_path / "dev.jsonl", "output_dir": tmp_path / "ckpt",
        "merged_dir": tmp_path / "merged", "results_dir": tmp_path / "res",
        "merge_results_dir": tmp_path / "mres", "attn_implementation": "eager", "smoke_clips": 1,
        "optim": base.optim.model_copy(update={"bf16": False, "gradient_checkpointing": False,
                                               "per_device_batch_size": 2,
                                               "dataloader_num_workers": 0}),
    })  # fmt: skip

    outcome = run_training(cfg, "tiny", smoke=True, record=record())
    assert outcome.lora_modules == 2 * 7  # two decoder layers x seven projections
    assert outcome.params.by_component["audio_tower"]["trainable"] == 0
    assert (tmp_path / "ckpt" / "tiny" / "best" / "adapter_config.json").exists()

    # A random model may emit nothing; the reload itself is what this test checks.
    engine_cfg = load_config(CONFIGS / "engines" / "hf_ft.yaml", EngineConfig).model_copy(
        update={"device": "cpu", "dtype": "float32", "max_new_tokens": 4}
    )
    real_make_engine = factory.make_engine
    reloaded: list[object] = []

    def make_and_check(ec: EngineConfig) -> ScriptedEngine:
        reloaded.append(real_make_engine(ec))  # loads the merged checkpoint for real
        return ScriptedEngine({entries[0].audio: "satu 0"})

    monkeypatch.setattr("asr_assess.inference.factory.make_engine", make_and_check)
    report = export.run_merge(cfg, engine_cfg, "tiny", record())
    assert report.ok
    assert not report.unexpected_changes
    assert len(report.changed) == 2 * 7
    assert reloaded
