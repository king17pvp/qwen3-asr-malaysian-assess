"""Markdown tables built only from load-test result files (results/loadtest/*/summary.json)."""

from collections.abc import Sequence
from pathlib import Path

from asr_assess.core.load_results import LevelResult, LoadRunSummary, read_summary

_DASH = "—"


def load_runs(root: Path) -> list[LoadRunSummary]:
    """Every run under ``root``, oldest first (by the run record's timestamp)."""
    runs = [read_summary(path) for path in sorted(root.glob("*/summary.json"))]
    return sorted(runs, key=lambda r: str(r.record.get("timestamp", "")))


def journey_table(runs: Sequence[LoadRunSummary]) -> str:
    """One row per run: max sustainable streams per threshold, and P95/VRAM/WER at the max."""
    thresholds = [v.threshold for v in runs[0].verdicts] if runs else []
    header = [
        "Configuration",
        "Profile",
        *(f"Max @{t:g}" for t in thresholds),
        "P95 RTF at max",
        "Peak VRAM GiB",
        "WER at max",
    ]
    rows = []
    for run in runs:
        best = run.verdicts[0].max_sustainable if run.verdicts else None
        at_max = next((lv for lv in run.levels if lv.level == best), None)
        rows.append(
            [
                run.label,
                run.profile,
                *(_opt(v.max_sustainable) for v in run.verdicts),
                _p95(at_max),
                _vram(at_max),
                _pct(at_max.wer if at_max else None),
            ]
        )
    return _markdown(header, rows)


def concurrency_table(run: LoadRunSummary) -> str:
    """One row per tested level of one run (streams, or live speakers for open loop)."""
    header = [
        "Live speakers" if run.mode == "open" else "Streams",
        "Avg RTF",
        "P50",
        "P95",
        "GPU util %",
        "VRAM GiB",
        "Audio-s/s",
        "Req/s",
        "WER",
    ]
    rows = [_level_row(lv) for lv in sorted(run.levels, key=lambda lv: lv.level)]
    return _markdown(header, rows)


def _level_row(lv: LevelResult) -> list[str]:
    rtf = lv.rtf
    util = f"{lv.gpu.mean_util_pct:.0f}" if lv.gpu else _DASH
    return [
        str(lv.level),
        _num(rtf.mean if rtf else None),
        _num(rtf.p50 if rtf else None),
        _p95(lv),
        util,
        _vram(lv),
        f"{lv.audio_s_per_s:.1f}",
        f"{lv.requests_per_s:.2f}",
        _pct(lv.wer),
    ]


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _opt(value: int | None) -> str:
    return _DASH if value is None else str(value)


def _num(value: float | None) -> str:
    return _DASH if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return _DASH if value is None else f"{value * 100:.1f}%"


def _p95(lv: LevelResult | None) -> str:
    return _num(lv.rtf.p95 if lv and lv.rtf else None)


def _vram(lv: LevelResult | None) -> str:
    return f"{lv.gpu.peak_mem_gib:.1f}" if lv and lv.gpu else _DASH
