"""Tests for publishing a verified merge to the Hugging Face Hub (no network: a recording fake)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from asr_assess.core.config import LoraTrainConfig, load_config
from asr_assess.training import publish
from tests.training.test_export import record

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SMOKE = {"id": "a.wav", "reference": "satu dua", "hypothesis": "satu dua", "language": "ms"}


@dataclass
class FakeCommit:
    oid: str
    commit_url: str


class FakeHub:
    """Records every Hub call; ``user`` None behaves like a machine with no token."""

    def __init__(self, user: str | None = "me") -> None:
        self.user = user
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def whoami(self) -> dict[str, Any]:
        if self.user is None:
            raise OSError("Invalid user token.")
        return {"name": self.user}

    def create_repo(self, **kwargs: Any) -> str:
        self.calls.append(("create_repo", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}"

    def upload_folder(self, **kwargs: Any) -> FakeCommit:
        self.calls.append(("upload_folder", kwargs))
        return FakeCommit("sha-weights", "https://huggingface.co/me/m/commit/sha-weights")

    def upload_file(self, **kwargs: Any) -> FakeCommit:
        self.calls.append(("upload_file", kwargs))
        return FakeCommit("sha-card", "https://huggingface.co/me/m/commit/sha-card")


def config(tmp_path: Path) -> LoraTrainConfig:
    return load_config(CONFIGS / "lora.yaml", LoraTrainConfig).model_copy(update={
        "merged_dir": tmp_path / "merged", "merge_results_dir": tmp_path / "res",
        "push_results_dir": tmp_path / "push",
    })  # fmt: skip


def write_merge(tmp_path: Path, ok: bool = True) -> None:
    """A merged checkpoint and the results `asr-assess merge` leaves next to it."""
    (tmp_path / "merged" / "r1").mkdir(parents=True)
    (tmp_path / "merged" / "r1" / "config.json").write_text("{}", encoding="utf-8")
    res = tmp_path / "res" / "r1"
    res.mkdir(parents=True)
    (res / "weight_deltas.json").write_text(json.dumps({"ok": ok}), encoding="utf-8")
    summary = {"record": record().model_dump(mode="json"), "changed_tensors": 196}
    (res / "merge_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (res / "smoke.jsonl").write_text(json.dumps(SMOKE) + "\n", encoding="utf-8")


def write_metrics(path: Path) -> Path:
    rate = {"value": 0.123, "low": 0.1, "high": 0.15}
    overall = {"n": 40, "audio_s": 300.0, "wer": rate, "cer": {**rate, "value": 0.05}}
    path.mkdir(parents=True)
    body = {"engine": "hf:ft", "n": 40, "report": {"overall": overall}}
    (path / "metrics.json").write_text(json.dumps(body), encoding="utf-8")
    return path / "metrics.json"


def front_matter(card: str) -> dict[str, Any]:
    _, meta, _ = card.split("---\n", 2)
    loaded: dict[str, Any] = yaml.safe_load(meta)
    return loaded


class TestVerifiedMerge:
    def test_missing_merge_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="asr-assess merge"):
            publish.verified_merge(config(tmp_path), "r1")

    def test_failed_weight_check_is_refused(self, tmp_path: Path) -> None:
        write_merge(tmp_path, ok=False)
        with pytest.raises(RuntimeError, match="weight_deltas"):
            publish.verified_merge(config(tmp_path), "r1")

    def test_returns_summary_and_smoke_rows(self, tmp_path: Path) -> None:
        write_merge(tmp_path)
        merge = publish.verified_merge(config(tmp_path), "r1")
        assert merge.checkpoint == tmp_path / "merged" / "r1"
        assert merge.summary["changed_tensors"] == 196
        assert merge.smoke == [SMOKE]


class TestModelCard:
    def card(self, tmp_path: Path, evals: list[Path] | None = None) -> str:
        write_merge(tmp_path)
        cfg = config(tmp_path)
        return publish.build_model_card(cfg, publish.verified_merge(cfg, "r1"), "me/m", evals or [])

    def test_hub_metadata(self, tmp_path: Path) -> None:
        meta = front_matter(self.card(tmp_path))
        assert meta["base_model"] == "Qwen/Qwen3-ASR-1.7B-hf"
        assert meta["pipeline_tag"] == "automatic-speech-recognition"
        assert meta["library_name"] == "transformers"
        assert meta["language"] == ["ms", "en"]

    def test_body_traces_lora_provenance_and_smoke(self, tmp_path: Path) -> None:
        card = self.card(tmp_path)
        assert "rank 16" in card and "alpha 32" in card
        assert "abc" in card  # the training commit
        assert "196" in card  # changed tensors
        assert "satu dua" in card
        assert "vllm serve me/m" in card

    def test_eval_metrics_become_a_wer_table(self, tmp_path: Path) -> None:
        metrics = write_metrics(tmp_path / "eval" / "hf_ft-eval")
        card = self.card(tmp_path, [metrics])
        assert "| hf_ft-eval | 40 | 12.30% [10.00, 15.00] | 5.00% |" in card

    def test_no_eval_metrics_means_no_table(self, tmp_path: Path) -> None:
        assert "WER" not in self.card(tmp_path)


class TestRunPush:
    def test_uploads_checkpoint_then_card_and_records_the_push(self, tmp_path: Path) -> None:
        write_merge(tmp_path)
        hub = FakeHub()
        summary = publish.run_push(config(tmp_path), "r1", "me/m", False, [], record(), hub)
        names = [name for name, _ in hub.calls]
        assert names == ["create_repo", "upload_folder", "upload_file"]
        create, folder, card = (kwargs for _, kwargs in hub.calls)
        assert create == {"repo_id": "me/m", "private": True, "exist_ok": True}
        assert folder["folder_path"] == str(tmp_path / "merged" / "r1")
        assert "abc" in folder["commit_message"]
        assert card["path_in_repo"] == "README.md"
        assert summary["commit"] == "sha-card"
        written = json.loads((tmp_path / "push" / "r1" / "push_summary.json").read_text("utf-8"))
        assert written["repo_id"] == "me/m"
        assert written["private"] is True
        assert written["weights_commit"] == "sha-weights"
        assert written["record"]["git_commit"] == "abc"

    def test_public_flag(self, tmp_path: Path) -> None:
        write_merge(tmp_path)
        hub = FakeHub()
        publish.run_push(config(tmp_path), "r1", "me/m", True, [], record(), hub)
        assert hub.calls[0][1]["private"] is False

    def test_no_token_fails_before_any_upload(self, tmp_path: Path) -> None:
        write_merge(tmp_path)
        hub = FakeHub(user=None)
        with pytest.raises(PermissionError, match="HF_TOKEN"):
            publish.run_push(config(tmp_path), "r1", "me/m", False, [], record(), hub)
        assert hub.calls == []

    def test_refuses_existing_push_record(self, tmp_path: Path) -> None:
        write_merge(tmp_path)
        (tmp_path / "push" / "r1").mkdir(parents=True)
        hub = FakeHub()
        with pytest.raises(FileExistsError, match="never overwritten"):
            publish.run_push(config(tmp_path), "r1", "me/m", False, [], record(), hub)
        assert hub.calls == []
