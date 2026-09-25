"""Tests for the JSONL manifest schema and read/write."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from asr_assess.core.manifest import ManifestEntry, read_manifest, write_manifest


def make_entry(**overrides: object) -> ManifestEntry:
    fields: dict[str, object] = {
        "audio": "wavs/utt0001.wav",
        "text": "language Malay<asr_text>apa khabar",
        "duration": 7.3,
        "bucket": "5-15",
        "source": "google/fleurs:ms_my",
    }
    fields.update(overrides)
    return ManifestEntry.model_validate(fields)


class TestSchema:
    def test_accepts_spec_example(self) -> None:
        entry = make_entry()
        assert entry.language == "Malay"
        assert entry.transcript == "apa khabar"

    def test_language_none_for_mixed_clips(self) -> None:
        assert make_entry(text="language None<asr_text>boleh lah bro").language is None

    def test_rejects_text_without_qwen_prefix(self) -> None:
        with pytest.raises(ValidationError, match="asr_text"):
            make_entry(text="apa khabar")

    def test_rejects_nonpositive_duration(self) -> None:
        with pytest.raises(ValidationError):
            make_entry(duration=0.0)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            make_entry(speaker="x")

    def test_is_immutable(self) -> None:
        entry = make_entry()
        with pytest.raises(ValidationError):
            entry.duration = 1.0  # type: ignore[misc]


class TestReadWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "train.jsonl"
        entries = [make_entry(), make_entry(audio="wavs/b.wav", bucket="2-5", duration=3.0)]

        write_manifest(path, entries)

        assert read_manifest(path) == entries

    def test_one_json_object_per_line_in_spec_key_order(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        write_manifest(path, [make_entry()])

        lines = path.read_text(encoding="utf-8").splitlines()

        assert len(lines) == 1
        assert list(json.loads(lines[0])) == ["audio", "text", "duration", "bucket", "source"]

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(make_entry().model_dump_json() + "\n\n", encoding="utf-8")
        assert len(read_manifest(path)) == 1

    def test_read_reports_bad_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(make_entry().model_dump_json() + '\n{"audio": "x"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"m\.jsonl:2"):
            read_manifest(path)
