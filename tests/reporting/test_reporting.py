"""Tests for tables and plots generated from load-test result files."""

from pathlib import Path

from typer.testing import CliRunner

from asr_assess.cli import app
from asr_assess.core.load_results import (
    GpuWindow,
    LevelResult,
    LoadRunSummary,
    RTFSummary,
    Verdict,
)
from asr_assess.reporting.plots import plot_p95, plot_throughput
from asr_assess.reporting.tables import concurrency_table, journey_table, load_runs


def level(n: int, p95: float, wer: float | None = 0.1) -> LevelResult:
    rtf = RTFSummary(count=10, mean=p95 / 2, p50=p95 / 2, p95=p95, p99=p95, max=p95)
    gpu = GpuWindow(mean_util_pct=50.0, peak_util_pct=90.0, mean_mem_gib=5.0, peak_mem_gib=6.5)
    return LevelResult(
        level=n,
        repeats=1,
        n_sent=10,
        n_ok=10,
        n_timeout=0,
        n_error=0,
        rtf=rtf,
        requests_per_s=float(n),
        audio_s_per_s=5.0 * n,
        wer=wer,
        cer=0.05,
        gpu=gpu,
        vllm=None,
    )


def summary(label: str, stamp: str, levels: list[LevelResult], best: int | None) -> LoadRunSummary:
    return LoadRunSummary(
        label=label,
        profile="quick",
        url="http://s",
        server_config=None,
        record={"timestamp": stamp},
        env={},
        reference_wer=0.1,
        levels=levels,
        verdicts=[
            Verdict(threshold=0.5, max_sustainable=best),
            Verdict(threshold=0.3, max_sustainable=None),
        ],
    )


def write_runs(root: Path) -> None:
    runs = [
        summary("vllm", "2026-09-29T10:00:00Z", [level(1, 0.1), level(2, 0.2), level(4, 0.4)], 4),
        summary("baseline", "2026-09-29T09:00:00Z", [level(1, 0.6)], None),
    ]
    for run in runs:
        (root / run.label).mkdir(parents=True)
        (root / run.label / "summary.json").write_text(run.model_dump_json(), encoding="utf-8")


def test_runs_load_in_timestamp_order(tmp_path: Path) -> None:
    write_runs(tmp_path)
    assert [r.label for r in load_runs(tmp_path)] == ["baseline", "vllm"]


def test_journey_table_row_per_run_with_dash_when_nothing_sustains(tmp_path: Path) -> None:
    write_runs(tmp_path)
    lines = journey_table(load_runs(tmp_path)).splitlines()
    assert lines[0].startswith("| Configuration |")
    baseline, vllm = lines[2], lines[3]
    assert baseline.startswith("| baseline | quick | — |")
    assert vllm.startswith("| vllm | quick | 4 | — | 0.400 | 6.5 | 10.0% |")


def test_concurrency_table_lists_levels_in_order(tmp_path: Path) -> None:
    write_runs(tmp_path)
    run = next(r for r in load_runs(tmp_path) if r.label == "vllm")
    rows = concurrency_table(run).splitlines()[2:]
    assert [row.split("|")[1].strip() for row in rows] == ["1", "2", "4"]
    assert rows[2].startswith("| 4 | 0.200 | 0.200 | 0.400 | 50 | 6.5 | 20.0 | 4.00 | 10.0% |")


def test_plots_write_pngs(tmp_path: Path) -> None:
    write_runs(tmp_path)
    runs = load_runs(tmp_path)
    plot_p95(runs, tmp_path / "p95.png", thresholds=[0.5, 0.3])
    plot_throughput(runs, tmp_path / "tp.png")
    for name in ("p95.png", "tp.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_report_command_writes_tables_and_plots(tmp_path: Path) -> None:
    write_runs(tmp_path / "loadtest")
    out = tmp_path / "plots"
    args = ["report", "--results", str(tmp_path / "loadtest"), "--out", str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    expected = {
        "journey.md",
        "concurrency_baseline.md",
        "concurrency_vllm.md",
        "p95_rtf.png",
        "throughput.png",
    }
    assert expected <= {p.name for p in out.iterdir()}
