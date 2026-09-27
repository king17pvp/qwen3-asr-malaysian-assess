"""`asr-assess` command line: thin commands that parse config and call library functions.

Heavy dependencies (torch, transformers, vllm) are imported inside the commands that need them.
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from asr_assess.core.config import (
    BenchConfig,
    DataConfig,
    EngineConfig,
    EvalConfig,
    LoadTestConfig,
    LoraTrainConfig,
    StrictModel,
    load_config,
)
from asr_assess.core.logging import setup_logging
from asr_assess.core.manifest import read_manifest
from asr_assess.core.run_record import collect_run_record

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


CONFIG_MODELS: dict[ConfigKind, type[StrictModel]] = {
    ConfigKind.data: DataConfig,
    ConfigKind.lora: LoraTrainConfig,
    ConfigKind.loadtest: LoadTestConfig,
    ConfigKind.engine: EngineConfig,
    ConfigKind.eval: EvalConfig,
    ConfigKind.bench: BenchConfig,
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
        load_config(path, CONFIG_MODELS[kind])
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


INFERENCE_PACKAGES = ["torch", "transformers", "numpy", "jiwer", "soundfile"]


class EvalRun(StrictModel):
    """Everything that determines an eval result; stamped into metrics.json."""

    engine: EngineConfig
    eval: EvalConfig
    manifest: str
    limit: int | None


class BenchRun(StrictModel):
    """Everything that determines an offline benchmark result; stamped into summary.json."""

    engine: EngineConfig
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
        engine=load_config(engine, EngineConfig),
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
    run = BenchRun(engine=load_config(engine, EngineConfig), bench=bench_cfg, smoke=smoke)
    out_dir = bench_cfg.output_dir / (run_name or f"{engine.stem}-offline")
    record = collect_run_record(run, packages=INFERENCE_PACKAGES, repo=Path.cwd())
    env = collect_env_info(INFERENCE_PACKAGES)
    asr = factory.make_engine(run.engine)
    entries = read_manifest(bench_cfg.manifest)
    run_offline(asr, entries, bench_cfg, run.engine.language_hint, out_dir, record, env)


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
    engine: EngineConfig
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

    cfg, engine_cfg = load_config(config, LoraTrainConfig), load_config(engine, EngineConfig)
    name = run_name or config.stem
    run = MergeRun(lora=cfg, engine=engine_cfg, run_name=name)
    record = collect_run_record(run, TRAIN_PACKAGES, Path.cwd())
    export.run_merge(cfg, engine_cfg, name, record)
