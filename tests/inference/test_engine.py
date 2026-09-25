"""Tests for the engine contract."""

import numpy as np
import pytest

from asr_assess.inference.engine import ASREngine, AudioRequest, Transcript, check_alignment
from tests.fakes import ScriptedEngine

SILENCE = np.zeros(16000, dtype=np.float32)


def batch(*ids: str) -> list[AudioRequest]:
    return [AudioRequest(i, SILENCE) for i in ids]


def test_scripted_engine_satisfies_the_protocol() -> None:
    engine: ASREngine = ScriptedEngine({"a": "hello"})
    assert engine.transcribe(batch("a")) == [Transcript("a", "hello", None)]


def test_alignment_accepts_matching_ids() -> None:
    check_alignment(batch("a", "b"), [Transcript("a", "", None), Transcript("b", "", None)])


def test_alignment_rejects_reordered_output() -> None:
    with pytest.raises(RuntimeError, match="ids"):
        check_alignment(batch("a", "b"), [Transcript("b", "", None), Transcript("a", "", None)])


def test_alignment_rejects_missing_output() -> None:
    with pytest.raises(RuntimeError, match="ids"):
        check_alignment(batch("a", "b"), [Transcript("a", "", None)])
