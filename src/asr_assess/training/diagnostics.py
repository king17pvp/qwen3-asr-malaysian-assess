"""Evidence for the report: how many parameters train, and which weights actually changed."""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_COMPONENTS = ("audio_tower", "multi_modal_projector", "language_model")


@dataclass(frozen=True)
class ParamCount:
    """Total and trainable parameters, overall and per model component."""

    total: int
    trainable: int
    by_component: dict[str, dict[str, int]]

    @property
    def trainable_pct(self) -> float:
        """Share of parameters that train, in percent."""
        return 100.0 * self.trainable / self.total if self.total else 0.0


def count_parameters(params: Iterable[tuple[str, int, bool]]) -> ParamCount:
    """Count ``(name, numel, requires_grad)`` triples, e.g. from ``model.named_parameters()``."""
    by_component: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, numel, requires_grad in params:
        bucket = by_component[_component(name)]
        bucket["total"] += numel
        bucket["trainable"] += numel if requires_grad else 0
    total = sum(b["total"] for b in by_component.values())
    trainable = sum(b["trainable"] for b in by_component.values())
    return ParamCount(total, trainable, dict(sorted(by_component.items())))


def _component(name: str) -> str:
    return next((c for c in _COMPONENTS if f".{c}." in f".{name}"), "other")


def relative_delta(base: np.ndarray, tuned: np.ndarray) -> float:
    """‖tuned − base‖_F / ‖base‖_F (the absolute norm when ``base`` is all zeros)."""
    base64 = base.astype(np.float64)
    diff = float(np.linalg.norm(tuned.astype(np.float64) - base64))
    norm = float(np.linalg.norm(base64))
    return diff / norm if norm else diff


@dataclass(frozen=True)
class DeltaReport:
    """Which tensors changed after merging, and whether only LoRA targets did."""

    changed: dict[str, float]
    unchanged: int
    unexpected_changes: list[str]
    unchanged_targets: list[str]

    @property
    def ok(self) -> bool:
        """Some target changed and nothing outside the targets did."""
        return bool(self.changed) and not self.unexpected_changes


def summarize_deltas(deltas: Mapping[str, float], target_pattern: str) -> DeltaReport:
    """Split per-tensor deltas into expected changes, unexpected changes and untouched targets."""

    def is_target(tensor: str) -> bool:
        return re.fullmatch(target_pattern, tensor.rsplit(".", 1)[0]) is not None

    changed = {name: d for name, d in sorted(deltas.items()) if d > 0.0}
    return DeltaReport(
        changed=changed,
        unchanged=len(deltas) - len(changed),
        unexpected_changes=[name for name in changed if not is_target(name)],
        unchanged_targets=[n for n, d in sorted(deltas.items()) if d == 0.0 and is_target(n)],
    )


def weight_deltas(base_dir: Path, tuned_dir: Path) -> dict[str, float]:
    """Relative change of every tensor, streamed one at a time from both checkpoints on disk."""
    from safetensors import safe_open

    with ExitStack() as stack:
        base = _open_all(stack, safe_open, base_dir)
        tuned = _open_all(stack, safe_open, tuned_dir)
        if set(base) != set(tuned):
            only = sorted(set(base) ^ set(tuned))[:5]
            raise ValueError(f"Checkpoints hold different tensors, e.g. {only}")
        return {name: _delta(base[name].get_tensor(name), tuned[name].get_tensor(name))
                for name in sorted(base)}  # fmt: skip


def _open_all(stack: ExitStack, safe_open: Any, directory: Path) -> dict[str, Any]:
    """Tensor name -> the open safetensors handle that holds it (sharded or not)."""
    handles: dict[str, Any] = {}
    for path in sorted(directory.glob("*.safetensors")):
        handle = stack.enter_context(safe_open(str(path), framework="pt"))
        handles.update(dict.fromkeys(handle.keys(), handle))
    return handles


def _delta(base: Any, tuned: Any) -> float:
    # Bitwise equality first: frozen tensors (incl. the 311M-element embedding) skip the maths.
    if base.shape == tuned.shape and bool((base == tuned).all()):
        return 0.0
    return relative_delta(base.float().numpy(), tuned.float().numpy())
