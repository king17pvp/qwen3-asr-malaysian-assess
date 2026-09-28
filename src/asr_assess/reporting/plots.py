"""PNG plots built only from load-test result files (needs the `report` extra)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asr_assess.core.load_results import LoadRunSummary


def plot_p95(runs: Sequence[LoadRunSummary], path: Path, thresholds: Sequence[float]) -> None:
    """P95 RTF against concurrent streams, one line per run, dashed threshold lines."""
    fig, ax = _axes()
    for run in runs:
        points = [(lv.level, lv.rtf.p95) for lv in _ordered(run) if lv.rtf]
        if points:
            levels, p95s = zip(*points, strict=True)
            ax.plot(levels, p95s, marker="o", label=run.label)
    for threshold in thresholds:
        ax.axhline(threshold, linestyle="--", linewidth=1, color="grey")
    _finish(fig, ax, "P95 RTF", path)


def plot_throughput(runs: Sequence[LoadRunSummary], path: Path) -> None:
    """Audio seconds served per second against concurrent streams, one line per run."""
    fig, ax = _axes()
    for run in runs:
        levels = _ordered(run)
        ax.plot(
            [lv.level for lv in levels],
            [lv.audio_s_per_s for lv in levels],
            marker="o",
            label=run.label,
        )
    _finish(fig, ax, "Audio seconds per second", path)


def _ordered(run: LoadRunSummary) -> list[Any]:
    return sorted(run.levels, key=lambda lv: lv.level)


def _axes() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(7, 4.5))


def _finish(fig: Any, ax: Any, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent streams")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
