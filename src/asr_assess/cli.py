"""`asr-assess` command line: thin commands that parse config and call library functions.

Heavy dependencies (torch, transformers, vllm) are imported inside the commands that need them.
"""

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError

from asr_assess.core.config import (
    BenchConfig,
    DataConfig,
    EvalConfig,
    HFEngineConfig,
    LoadTestConfig,
    LoraTrainConfig,
    ServeConfig,
    StrictModel,
    VLLMHTTPEngineConfig,
    VLLMServeConfig,
    load_config,
    load_engine_config,
)
from asr_assess.core.logging import setup_logging
from asr_assess.core.manifest import read_manifest
from asr_assess.core.run_record import collect_run_record

if TYPE_CHECKING:
    from asr_assess.benchmark.gpu_monitor import GpuMonitor
    from asr_assess.benchmark.load_stats import PoolClip
    from asr_assess.benchmark.loadtest import RunMeta

log = logging.getLogger(__name__)

app = typer.Typer(
    help="Fine-tune and benchmark Qwen3-ASR-1.7B on Malaysian speech.",
    no_args_is_help=True,
)


class ConfigKind(StrEnum):
    """Config files that `check-config` can validate."""

    data = "data"
    lora = "lora"
    loadtest = "loadtest"
    engine = "engine"
    eval = "eval"
    bench = "bench"
    serve = "serve"
    vllm = "vllm"


CONFIG_LOADERS: dict[ConfigKind, Callable[[Path], object]] = {
    ConfigKind.data: partial(load_config, model=DataConfig),
    ConfigKind.lora: partial(load_config, model=LoraTrainConfig),
    ConfigKind.loadtest: partial(load_config, model=LoadTestConfig),
    ConfigKind.engine: load_engine_config,
    ConfigKind.eval: partial(load_config, model=EvalConfig),
    ConfigKind.bench: partial(load_config, model=BenchConfig),
    ConfigKind.serve: partial(load_config, model=ServeConfig),
    ConfigKind.vllm: partial(load_config, model=VLLMServeConfig),
}
DATA_PACKAGES = ["numpy", "soundfile", "librosa", "huggingface-hub", "pyarrow"]


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Configure logging for every subcommand."""
    setup_logging(log_level)


@app.command("check-config")
def check_config(
    kind: Annotated[ConfigKind, typer.Argument(help="Which config schema to validate against.")],
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML file.")],
) -> None:
    """Validate a YAML config file without running anything."""
    try:
        CONFIG_LOADERS[kind](path)
    except (ValidationError, ValueError) as err:
        log.error("%s is not a valid %s config:\n%s", path, kind.value, err)
        raise typer.Exit(code=1) from err
    log.info("%s is a valid %s config", path, kind.value)


@app.command()
def data(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("configs/data.yaml"),
    dry_run: Annotated[
        bool, typer.Option(help="Read metadata and plan the splits; fetch and write no audio.")
    ] = False,
) -> None:
    """Build the train/eval/control manifests from the Hub (needs the `data` extra)."""
    from asr_assess.data.build import build_dataset
    from asr_assess.data.sources import make_source

    cfg = load_config(config, DataConfig)
    try:
        sources = {name: make_source(cfg.sources[name]) for name in cfg.sources_in_use()}
    except ImportError as err:
        log.error("Missing dataset dependencies; run `uv sync --extra data` (%s)", err)
        raise typer.Exit(code=1) from err
    record = collect_run_record(cfg, packages=DATA_PACKAGES, repo=Path.cwd())
    build_dataset(cfg, sources, record, dry_run=dry_run)


@app.command("vllm-args")
def vllm_args_command(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="configs/vllm/*.yaml")],
) -> None:
    """Print `vllm serve` arguments for a vLLM config, one per line (used by vllm_serve.sh)."""
    from asr_assess.serving.vllm_args import vllm_args

    for arg in vllm_args(load_config(path, VLLMServeConfig)):
        typer.echo(arg)


INFERENCE_PACKAGES = ["torch", "transformers", "numpy", "jiwer", "soundfile"]


class EvalRun(StrictModel):
    """Everything that determines an eval result; stamped into metrics.json."""

    engine: HFEngineConfig | VLLMHTTPEngineConfig
    eval: EvalConfig
    manifest: str
    limit: int | None


class BenchRun(StrictModel):
    """Everything that determines an offline benchmark result; stamped into summary.json."""

    engine: HFEngineConfig | VLLMHTTPEngineConfig
    bench: BenchConfig
    smoke: bool


EngineOption = Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Engine YAML.")]


@app.command("eval")
def eval_command(
    engine: EngineOption,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("configs/eval.yaml"),
    manifest: Annotated[str, typer.Option(help="Key in the config's manifests.")] = "eval",
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N clips.")] = None,
    run_name: Annotated[str | None, typer.Option(help="Default: <engine>-<manifest>.")] = None,
) -> None:
    """WER/CER of an engine over a manifest (needs the `train` extra for the HF engine)."""
    from asr_assess.evaluation.evaluate import evaluate
    from asr_assess.inference import factory

    run = EvalRun(
        engine=load_engine_config(engine),
        eval=load_config(config, EvalConfig),
        manifest=manifest,
        limit=limit,
    )
    if manifest not in run.eval.manifests:
        log.error("Unknown manifest %r; choose from %s", manifest, sorted(run.eval.manifests))
        raise typer.Exit(code=1)
    entries = read_manifest(run.eval.manifests[manifest])[:limit]
    out_dir = run.eval.output_dir / (run_name or f"{engine.stem}-{manifest}")
    record = collect_run_record(run, packages=INFERENCE_PACKAGES, repo=Path.cwd())
    asr = factory.make_engine(run.engine)
    evaluate(asr, entries, run.eval, run.engine.language_hint, out_dir, record)


@app.command()
def bench(
    engine: EngineOption,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("configs/bench.yaml"),
    smoke: Annotated[bool, typer.Option(help="2 clips per bucket, 1 repeat, 1 warm-up.")] = False,
    run_name: Annotated[str | None, typer.Option(help="Default: <engine>-offline.")] = None,
) -> None:
    """Single-stream RTF per duration bucket (needs the `train` extra for the HF engine)."""
    from asr_assess.benchmark.env_info import collect_env_info
    from asr_assess.benchmark.offline import run_offline
    from asr_assess.inference import factory

    bench_cfg = load_config(config, BenchConfig)
    if smoke:
        bench_cfg = bench_cfg.model_copy(
            update={"clips_per_bucket": 2, "repeats": 1, "warmup_requests": 1}
        )
    run = BenchRun(engine=load_engine_config(engine), bench=bench_cfg, smoke=smoke)
    out_dir = bench_cfg.output_dir / (run_name or f"{engine.stem}-offline")
    record = collect_run_record(run, packages=INFERENCE_PACKAGES, repo=Path.cwd())
    env = collect_env_info(INFERENCE_PACKAGES)
    asr = factory.make_engine(run.engine)
    entries = read_manifest(bench_cfg.manifest)
    run_offline(asr, entries, bench_cfg, run.engine.language_hint, out_dir, record, env)


@app.command()
def serve(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="configs/serve/*.yaml")],
    dry_run: Annotated[bool, typer.Option(help="Validate the configs; load no model.")] = False,
) -> None:
    """HF-engine transcription server with dynamic batching (needs `train` + `http` extras)."""
    cfg = load_config(config, ServeConfig)
    engine_cfg = load_engine_config(cfg.engine)
    if not isinstance(engine_cfg, HFEngineConfig):
        log.error("serve runs HF engines only; vLLM is served by scripts/vllm_serve.sh")
        raise typer.Exit(code=1)
    log.info(
        "Serving %s (%s) on %s:%d, max_batch=%d, max_wait_ms=%s",
        engine_cfg.model_id,
        engine_cfg.attn_implementation,
        cfg.host,
        cfg.port,
        cfg.max_batch,
        cfg.max_wait_ms,
    )
    if dry_run:
        return
    import uvicorn

    from asr_assess.inference import factory
    from asr_assess.serving.app import create_app
    from asr_assess.serving.batcher import Batcher

    batcher = Batcher(factory.make_engine(engine_cfg), cfg.max_batch, cfg.max_wait_ms / 1000)
    uvicorn.run(create_app(batcher, cfg.sample_rate), host=cfg.host, port=cfg.port)


LOADTEST_PACKAGES = ["httpx", "numpy", "soundfile", "pynvml"]


class LoadTestRun(StrictModel):
    """Everything that determines a load-test result; stamped into summary.json."""

    loadtest: LoadTestConfig
    profile: str
    url: str
    label: str
    max_tokens: int


@app.command()
def loadtest(
    url: Annotated[str, typer.Option(help="Server base URL, e.g. http://localhost:8000")],
    label: Annotated[
        str, typer.Option(help="Journey row name; results go to <output_dir>/<label>")
    ],
    profile: Annotated[str, typer.Option(help="Profile in the config: full or quick")] = "quick",
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/loadtest.yaml"
    ),
    server_config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Server YAML, copied into the summary"),
    ] = None,
    max_tokens: Annotated[int, typer.Option(min=1, help="Decode budget sent per request")] = 256,
    dry_run: Annotated[
        bool, typer.Option(help="Build the audio pool and log the schedule; send nothing.")
    ] = False,
) -> None:
    """Closed-loop N-stream load test against an HF or vLLM server (needs the `http` extra)."""
    from asr_assess.benchmark import loadtest as lt
    from asr_assess.benchmark.env_info import collect_env_info

    cfg = load_config(config, LoadTestConfig)
    if profile not in cfg.profiles:
        log.error("Unknown profile %r; choose from %s", profile, sorted(cfg.profiles))
        raise typer.Exit(code=1)
    out_dir = cfg.output_dir / label
    if out_dir.exists():
        log.error("%s exists; results are never overwritten", out_dir)
        raise typer.Exit(code=1)
    clips = lt.build_pool(read_manifest(cfg.audio_pool.manifest), sample_rate=16000)
    chosen = cfg.profiles[profile]
    worst_s = (
        len(chosen.concurrency_levels) * chosen.repeats * (chosen.warmup_s + chosen.steady_state_s)
    )
    log.info(
        "%d clips; levels %s x %d repeats; at most ~%.0f min before bisection",
        len(clips),
        chosen.concurrency_levels,
        chosen.repeats,
        worst_s / 60,
    )
    if dry_run:
        return
    run = LoadTestRun(loadtest=cfg, profile=profile, url=url, label=label, max_tokens=max_tokens)
    meta = lt.RunMeta(
        label=label,
        url=url,
        server_config=server_config.read_text(encoding="utf-8") if server_config else None,
        record=collect_run_record(run, packages=LOADTEST_PACKAGES, repo=Path.cwd()),
        env=collect_env_info(LOADTEST_PACKAGES),
    )
    asyncio.run(_run_loadtest(cfg, profile, out_dir, meta, clips, max_tokens))


async def _run_loadtest(
    cfg: LoadTestConfig,
    profile: str,
    out_dir: Path,
    meta: "RunMeta",
    clips: "list[PoolClip]",
    max_tokens: int,
) -> None:
    from asr_assess.benchmark.client import OpenAITranscriptionTransport
    from asr_assess.benchmark.loadtest import run_loadtest

    transport = OpenAITranscriptionTransport(meta.url, cfg.request_timeout_s, max_tokens)
    try:
        with _gpu_monitor(cfg.gpu_sample_hz) as gpu:
            summary = await run_loadtest(transport, clips, cfg, profile, out_dir, meta, gpu)
    finally:
        await transport.aclose()
    for verdict in summary.verdicts:
        log.info("Max sustainable at P95 RTF <= %s: %s", verdict.threshold, verdict.max_sustainable)


@contextmanager
def _gpu_monitor(hz: float) -> "Iterator[GpuMonitor | None]":
    """NVML sampling of GPU 0, or None (with a warning) where NVML is unavailable."""
    from asr_assess.benchmark.gpu_monitor import GpuMonitor, PynvmlReader

    try:
        reader = PynvmlReader()
    except Exception as err:  # no driver or no pynvml: measure without GPU stats
        log.warning("GPU sampling disabled: %s", err)
        yield None
        return
    with GpuMonitor(reader, hz) as monitor:
        yield monitor


TRAIN_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "numpy",
    "soundfile",
]
LoraOption = Annotated[Path, typer.Option(exists=True, dir_okay=False, help="LoRA YAML.")]


class TrainRun(StrictModel):
    """Everything that determines a training run; stamped into train_summary.json."""

    lora: LoraTrainConfig
    smoke: bool


class MergeRun(StrictModel):
    """Everything that determines a merge; stamped into merge_summary.json."""

    lora: LoraTrainConfig
    engine: HFEngineConfig
    run_name: str


@app.command()
def train(
    config: LoraOption = Path("configs/lora.yaml"),
    smoke: Annotated[
        bool, typer.Option(help="A few clips and steps (see `smoke` in the config).")
    ] = False,
    run_name: Annotated[
        str | None, typer.Option(help="Default: <config> or <config>-smoke.")
    ] = None,
) -> None:
    """Decoder-only LoRA fine-tuning; keeps the best epoch by dev loss (needs the `train` extra)."""
    from asr_assess.training import trainer

    cfg = load_config(config, LoraTrainConfig)
    name = run_name or (f"{config.stem}-smoke" if smoke else config.stem)
    record = collect_run_record(TrainRun(lora=cfg, smoke=smoke), TRAIN_PACKAGES, Path.cwd())
    trainer.run_training(cfg, name, smoke, record)


@app.command()
def merge(
    engine: EngineOption,
    config: LoraOption = Path("configs/lora.yaml"),
    run_name: Annotated[
        str | None, typer.Option(help="The training run. Default: <config>.")
    ] = None,
) -> None:
    """Merge a run's best adapter, verify the weight deltas, reload and transcribe dev clips."""
    from asr_assess.training import export

    cfg, engine_cfg = load_config(config, LoraTrainConfig), load_config(engine, HFEngineConfig)
    name = run_name or config.stem
    run = MergeRun(lora=cfg, engine=engine_cfg, run_name=name)
    record = collect_run_record(run, TRAIN_PACKAGES, Path.cwd())
    export.run_merge(cfg, engine_cfg, name, record)
