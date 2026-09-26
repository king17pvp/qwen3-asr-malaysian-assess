"""weight_deltas streams real safetensors files. Run with: uv run --extra train pytest -m hf"""

from pathlib import Path
from typing import Any

import pytest

from asr_assess.training.diagnostics import weight_deltas

pytestmark = pytest.mark.hf


def save(directory: Path, tensors: dict[str, Any], name: str = "model.safetensors") -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / name))


def test_streams_bf16_tensors_across_shards(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    same = torch.ones(4, 4, dtype=torch.bfloat16)
    save(tmp_path / "base", {"a": same, "b": same})
    save(tmp_path / "tuned", {"a": same}, "model-00001-of-00002.safetensors")
    save(tmp_path / "tuned", {"b": same * 2}, "model-00002-of-00002.safetensors")
    deltas = weight_deltas(tmp_path / "base", tmp_path / "tuned")
    assert deltas == {"a": 0.0, "b": pytest.approx(1.0)}


def test_mismatched_keys_raise(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save(tmp_path / "base", {"a": torch.ones(2)})
    save(tmp_path / "tuned", {"renamed.a": torch.ones(2)})
    with pytest.raises(ValueError, match="different tensors"):
        weight_deltas(tmp_path / "base", tmp_path / "tuned")
