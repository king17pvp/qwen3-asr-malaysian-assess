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
    DataConfig,
    LoadTestConfig,
    LoraTrainConfig,
    StrictModel,
    load_config,
)
from asr_assess.core.logging import setup_logging
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


CONFIG_MODELS: dict[ConfigKind, type[StrictModel]] = {
    ConfigKind.data: DataConfig,
    ConfigKind.lora: LoraTrainConfig,
    ConfigKind.loadtest: LoadTestConfig,
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
