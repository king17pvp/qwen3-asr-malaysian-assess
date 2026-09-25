"""Tests for turning manifest entries into engine requests."""

from pathlib import Path

import pytest

from asr_assess.inference.requests import to_requests
from tests.fakes import write_clips

SPECS = [("a", 3.0, "2-5", "satu dua"), ("b", 6.0, "5-15", "tiga empat")]


def test_requests_carry_manifest_language_only_when_asked(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    assert {r.language for r in to_requests(entries, 16000, "auto")} == {None}
    assert {r.language for r in to_requests(entries, 16000, "manifest")} == {"Malay"}


def test_missing_audio_names_the_file(tmp_path: Path) -> None:
    entries = write_clips(tmp_path, SPECS)
    Path(entries[0].audio).unlink()
    with pytest.raises(FileNotFoundError, match=r"a\.wav"):
        to_requests(entries, 16000, "auto")
