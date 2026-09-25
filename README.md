# qwen3-asr-malaysian-assess

Fine-tune Qwen3-ASR-1.7B on Malaysian speech and optimise it for high-concurrency inference on one GPU (8nabler ML Engineer assessment).

_Work in progress: this README will cover the uv quickstart, one command per result, and the hardware table._

## Data

```bash
uv sync --extra data                   # huggingface-hub + pyarrow
uv run asr-assess data --dry-run       # plan every split from metadata (~1 min, no audio)
uv run asr-assess data                 # write data/audio/**.wav + data/manifests/*.jsonl
```

Set `HF_TOKEN` in the environment to avoid Hub rate limits. Nothing is downloaded in full:
the build reads parquet columns/row groups and single zip members with HTTP range requests.

| Category | Train | Eval | Source | Held out by | Label |
|---|---|---|---|---|---|
| Manglish / code-switching | 20 min | 6 min | mesolitica/Malaysian-STT-Whisper `malaysian_context_v2` | YouTube video | `language None` |
| Malay conversational | 5 min | 2 min | mesolitica/Malaysian-STT-Whisper `extra` (Malay Conversational Speech Corpus) | speaker | `language Malay` |
| Malay read | 5 min | 2 min | FLEURS `ms_my` (train: validation, eval: test) | official split | `language Malay` |
| English read | 5 min | 2 min | FLEURS `en_us` (train: validation, eval: test) | official split | `language English` |
| Control | — | 5 min | LibriSpeech test-clean (never trained on) | — | `language English` |

Clips are 2–30 s, resampled to 16 kHz mono and spread over the 2–5 / 5–15 / 15–30 s buckets.
Sampling is seeded (`seed` in `configs/data.yaml`); `data/manifests/stats.json` records minutes per
split, bucket and source together with the git commit and config hash.

## Baseline (GPU)

The baseline is plain Transformers (`configs/engines/hf_base.yaml`: bf16, eager attention,
greedy decoding, batch 1). RTF = processing time / audio duration, measured end to end
(feature extraction + generate + decode) for one request at a time after 5 discarded warm-up
requests.

```bash
# on the GPU box
HF_TOKEN=... bash scripts/vast_setup.sh <repo-url>
# from your machine: use the exact dataset built locally
rsync -avz data/ <box>:~/qwen3-asr-malaysian-assess/data/
# on the box, in qwen3-asr-malaysian-assess/
uv run pytest -m model                      # real-model smoke test: run this first
uv run asr-assess eval --engine configs/engines/hf_base.yaml --manifest eval --limit 3 \
    --run-name smoke-eval                   # a 3-clip dry run of the full path
uv run asr-assess eval --engine configs/engines/hf_base.yaml --manifest eval      # results/eval/hf_base-eval/
uv run asr-assess eval --engine configs/engines/hf_base.yaml --manifest control   # results/eval/hf_base-control/
uv run asr-assess bench --engine configs/engines/hf_base.yaml                     # results/bench/hf_base-offline/
# back on your machine
rsync -avz <box>:~/qwen3-asr-malaysian-assess/results/ results/
```

`make eval-base` and `make bench-base` are shortcuts for the last three commands. Result
folders are never overwritten: re-runs need a new `--run-name`. `metrics.json` and
`summary.json` record the git commit, config, GPU and time of every run.

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
