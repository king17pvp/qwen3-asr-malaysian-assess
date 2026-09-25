"""Smoke test against the real checkpoint. Run with: uv run pytest -m model"""

from pathlib import Path

import pytest

from asr_assess.core.audio import load_audio
from asr_assess.core.config import EngineConfig
from asr_assess.core.manifest import read_manifest
from asr_assess.inference.engine import AudioRequest

pytestmark = pytest.mark.model
CONTROL = Path("data/manifests/control.jsonl")
MIN_GPU_GIB = 8


@pytest.fixture(scope="module")
def engine() -> object:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from asr_assess.inference.hf_engine import HFEngine

    # Small GPUs cannot hold the ~4.7 GB bf16 weights; bf16 also keeps CPU RAM under ~6 GB.
    big_gpu = (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory >= MIN_GPU_GIB * 1024**3
    )
    device = "cuda" if big_gpu else "cpu"
    return HFEngine(EngineConfig(kind="hf", model_id="Qwen/Qwen3-ASR-1.7B-hf", dtype="bfloat16",
                                 device=device, attn_implementation="eager",
                                 max_new_tokens=256, language_hint="auto"))  # fmt: skip


@pytest.mark.skipif(not CONTROL.exists(), reason="build the dataset first: asr-assess data")
def test_transcribes_1d_numpy_audio_in_order(engine: object) -> None:
    entries = read_manifest(CONTROL)[:2]
    batch = [AudioRequest(e.audio, load_audio(Path(e.audio), 16000)) for e in entries]
    out = engine.transcribe(batch)  # type: ignore[attr-defined]
    assert [t.id for t in out] == [e.audio for e in entries]
    for t in out:
        # If decode stripped <asr_text> as a special token, the prefix would leak into the text
        # and the language would be lost: every WER would silently include "language english".
        assert t.text and "<asr_text>" not in t.text
        assert not t.text.lower().startswith("language"), t.text
        assert t.language == "English", t  # LibriSpeech control clips are English speech
