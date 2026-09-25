# qwen3-asr-malaysian-assess

Fine-tune Qwen3-ASR-1.7B on Malaysian speech and optimise it for high-concurrency inference on one GPU (8nabler ML Engineer assessment).

_Work in progress: this README will cover the uv quickstart, one command per result, and the hardware table._

## Development

Everything runs through [uv](https://docs.astral.sh/uv/); the base install and CPU tests need no GPU.

```bash
uv sync --group dev             # base deps + ruff, mypy, pytest, pre-commit
uv run pre-commit install       # ruff + mypy on every commit
uv run pytest                   # CPU-only tests
uv run asr-assess --help
uv run asr-assess check-config lora configs/lora.yaml
```

`make install | test | lint | format | check-configs` are aliases for the same commands.

The `train` and `serve` extras conflict (vLLM pins its own torch/transformers) and are
installed separately on the GPU machine: `uv sync --extra train` or `uv sync --extra serve`.
