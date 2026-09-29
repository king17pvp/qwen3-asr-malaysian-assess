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

| Category | Train | Dev | Eval | Source | Held out by | Label |
|---|---|---|---|---|---|---|
| Manglish / code-switching | 20 min | 2 min | 6 min | mesolitica/Malaysian-STT-Whisper `malaysian_context_v2` | YouTube video | `language None` |
| Malay conversational | 5 min | 0.5 min | 2 min | mesolitica/Malaysian-STT-Whisper `extra` (Malay Conversational Speech Corpus) | speaker | `language Malay` |
| Malay read | 5 min | 0.5 min | 2 min | FLEURS `ms_my` (train/dev: validation, eval: test) | official split | `language Malay` |
| English read | 5 min | 0.5 min | 2 min | FLEURS `en_us` (train/dev: validation, eval: test) | official split | `language English` |
| Control | — | — | 5 min | LibriSpeech test-clean (never trained on) | — | `language English` |

Dev is used only to pick the best fine-tuning epoch, so the eval set stays unseen until the
final before/after comparison. It is drawn after train and eval from the train side, never
reuses a train clip, and avoids train's speakers/videos where any are left
(`dev_group_overlap` in `stats.json` flags the categories where none were).

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

## Fine-tuning (GPU)

LoRA (rank 16, alpha 32) on the Qwen3-1.7B decoder's attention and MLP projections only; the
audio encoder, projector and embeddings stay frozen. Five epochs, and the epoch with the
lowest **dev** loss is kept, so `eval.jsonl` is never seen before the final comparison.
Hyperparameters live in `configs/lora.yaml`.

```bash
# on the GPU box, after the baseline
make train-smoke   # 2 steps on 8 clips: catches API/OOM problems in about a minute
make train         # checkpoints/lora/lora/best/ + results/train/lora/{train_summary.json,log_history.jsonl}
make merge         # checkpoints/merged/lora/ + results/merge/lora/{weight_deltas.json,smoke.jsonl}
make eval-ft       # results/eval/hf_ft-eval/ and results/eval/hf_ft-control/
```

`merge` fails if any tensor outside the LoRA targets changed, or if none did. The fine-tuned
engine (`configs/engines/hf_ft.yaml`) is the baseline engine with only the weights swapped, so
before/after WER differs by fine-tuning alone.

Tests that need the `train` extra but no GPU (the real processor and a tiny random Qwen3-ASR
run through train → merge → reload): `uv run --extra train pytest -m hf`.

## Publishing to the Hugging Face Hub

`push` uploads a merged checkpoint that passed its weight-delta check, plus a generated model card
(base model, LoRA settings, training commit, smoke transcripts and, if given, WER/CER tables). The
repo is private unless `--public` is passed. Log in first with `hf auth login` or set `HF_TOKEN`.

```bash
make push REPO=<user>/qwen3-asr-malaysian   # + results/push/lora/push_summary.json (Hub commit)
# same, with public visibility and the fine-tuned WER in the card:
uv run --extra data asr-assess push --repo <user>/qwen3-asr-malaysian --public \
  --eval-metrics results/eval/hf_ft-eval/metrics.json \
  --eval-metrics results/eval/hf_ft-control/metrics.json
```

After that, `model_id: <user>/qwen3-asr-malaysian` in `configs/engines/hf_ft.yaml` (or
`vllm serve <user>/qwen3-asr-malaysian`) loads the model from the Hub instead of `checkpoints/`.

## Serving and load test (GPU)

Every journey row serves the fine-tuned merged checkpoint (`checkpoints/merged/lora`) behind the
same OpenAI-style endpoint, `POST /v1/audio/transcriptions` (multipart WAV → `{"text": ...}`):
either our HF server (`asr-assess serve`: FastAPI + a dynamic batcher in front of the
Transformers engine) or stock `vllm serve`. One load-test client drives both.

**RTF under load.** A *stream* is a closed-loop client: it sends one utterance, waits for the
transcript, then sends the next. Per-request RTF = end-to-end latency **including queueing** /
audio duration. A request counts when it is *sent* inside the steady-state window (warm-up is
discarded), so slow requests that finish after the window still count.

**Max sustainable** = the highest concurrency where P95 RTF ≤ 0.5, no request timed out or
failed, and WER is within 1 point of the same run's single-stream (level 1) WER, with every
lower tested level passing too. The ≤ 0.3 level is reported the same way. Bisection
refines the limit between the last passing and the first failing level.

| Process | Command | Extras |
|---|---|---|
| HF server | `make serve-hf CFG=configs/serve/<row>.yaml` (port 8001) | `train` + `http` |
| vLLM server | `make serve-vllm CFG=configs/vllm/<row>.yaml` (port 8000) | `serve` |
| Load-test client | `make loadtest URL=... LABEL=<row> PROFILE=quick\|full SERVER_CFG=...` | `http` (no torch) |
| Tables + plots | `make report` → `results/plots/` | `report` |

Run the server and the client in two terminals. One config file per row of the optimization
journey:

```bash
# HF rows (port 8001)
make serve-hf CFG=configs/serve/hf_baseline.yaml
make loadtest URL=http://localhost:8001 LABEL=hf-baseline PROFILE=full SERVER_CFG=configs/serve/hf_baseline.yaml
#   ... hf_sdpa.yaml (quick), hf_batched.yaml (quick; sweep max_batch / max_wait_ms)
# vLLM rows (port 8000)
make serve-vllm CFG=configs/vllm/default.yaml
make loadtest URL=http://localhost:8000 LABEL=vllm-default SERVER_CFG=configs/vllm/default.yaml
#   ... tuned.yaml, cpu_path.yaml, fp8.yaml, eager.yaml (quick); the best one again as `final` (full)
make report
```

Profiles live in `configs/loadtest.yaml`. `full` (3 repeats, 30 s warm-up, 120 s steady state)
is for the baseline and the final config, and `quick` (1 repeat, 15 s + 60 s) is for the
intermediate rows. Both stop one level after the first failing level. Results go to
`results/loadtest/<label>/` (`requests.jsonl`, `levels.jsonl`, `summary.json`), which is never
overwritten. `levels.jsonl` includes GPU utilization and VRAM (NVML at 10 Hz) and, for vLLM,
the running/waiting/KV-cache gauges scraped from `/metrics`.

vLLM notes:

- `bash scripts/vllm_serve.sh <yaml>` turns the YAML into `vllm serve` flags
  (`uv run asr-assess vllm-args <yaml>` prints them). vLLM 0.30's `Qwen3ASRForConditionalGeneration`
  maps the `-hf` weight layout and builds its config from the flat `-hf` `config.json`.
- Before any vLLM load test, check vLLM accuracy against HF on the same model:
  `uv run --extra http asr-assess eval --engine configs/engines/vllm_ft.yaml --manifest eval`
  vs `make eval-ft`. vLLM's transcription prompt leaves out the empty system turn that the HF
  chat template (and training) always has. TODO(gpu): record the WER gap here; above 1 point,
  switch the client to chat completions.
- The GPU is an RTX 3090 (Ampere). There, `quantization: fp8` is weight-only (W8A16) and gives
  no faster matmuls, so the FP8 row tests KV-cache capacity (`kv_cache_dtype: fp8`) rather than
  compute.

## Development

Everything runs through [uv](https://docs.astral.sh/uv/); the base install and CPU tests need no GPU.

```bash
uv sync --group dev --extra http --extra report   # base + ruff, mypy, pytest, pre-commit; as CI
uv run pre-commit install       # ruff + mypy on every commit
uv run pytest                   # CPU-only tests
uv run asr-assess --help
uv run asr-assess check-config lora configs/lora.yaml
```

`make install | test | lint | format | check-configs` are aliases for the same commands.

The `train` and `serve` extras conflict (vLLM pins its own torch/transformers) and are
installed separately on the GPU machine: `uv sync --extra train --extra http` (HF server) or
`uv sync --extra serve` (vLLM, includes `http`). The `http` extra (FastAPI, uvicorn, httpx,
pynvml) is pure Python and works with either.
