# Results before fine-tuning (baseline)

Off-the-shelf `Qwen/Qwen3-ASR-1.7B-hf` through plain Transformers (`configs/engines/hf_base.yaml`:
bf16, eager attention, greedy decoding, batch 1, `max_new_tokens` 256, no language hint).
These are the numbers the fine-tuned model and every inference optimization are compared against.

| | |
|---|---|
| Commit | `611456e` (clean tree) |
| GPU | NVIDIA GeForce RTX 3090, 24 GB (driver 580.126.09, CUDA runtime 12.6) |
| CPU / RAM | AMD Ryzen 7 5800X (16 threads), 62.7 GiB |
| Software | Python 3.12.3, torch 2.14.0+cu126, transformers 5.17.0, jiwer 4.0.0 |
| Data | `uv run asr-assess data` with the committed `configs/data.yaml` (seed 20260925) |
| Run date | 2026-09-26 |

Brackets are 95% bootstrap confidence intervals (1000 resamples, seed 0). WER/CER are computed
after the project normalizer (lowercase, punctuation stripped). No clip hit the token limit.

## Accuracy: held-out eval set (123 clips, 12.3 min)

| Slice | Clips | Audio | WER | CER |
|---|---:|---:|---|---|
| **Overall** | 123 | 736.5 s | **17.1%** [14.4, 20.6] | **7.5%** [6.0, 9.4] |
| Manglish / code-switching | 62 | 361.3 s | 18.8% [15.3, 23.2] | 8.6% [6.7, 11.2] |
| Malay conversational | 31 | 121.6 s | **30.6%** [22.0, 39.3] | 14.7% [10.4, 19.1] |
| Malay read (FLEURS) | 14 | 132.4 s | 10.5% [6.8, 16.4] | 3.1% [1.7, 4.9] |
| English read (FLEURS) | 16 | 121.3 s | 3.6% [1.2, 6.0] | 1.3% [0.3, 2.5] |

| Duration bucket | Clips | WER | CER |
|---|---:|---|---|
| 2–5 s | 74 | 23.2% [18.8, 28.8] | 11.4% [8.9, 14.3] |
| 5–15 s | 42 | 14.2% [10.0, 19.3] | 5.9% [3.9, 8.2] |
| 15–30 s | 7 | 11.7% [6.0, 18.0] | 4.2% [1.7, 7.9] |

## Accuracy: LibriSpeech test-clean control (44 clips, 5.0 min)

| Slice | Clips | WER | CER |
|---|---:|---|---|
| **Overall** | 44 | **1.7%** [0.6, 2.9] | **0.6%** [0.2, 1.1] |
| 2–5 s | 23 | 2.4% [0.5, 4.8] | 0.8% [0.0, 2.0] |
| 5–15 s | 17 | 1.3% [0.0, 2.8] | 0.7% [0.0, 1.4] |
| 15–30 s | 4 | 1.7% [0.0, 5.4] | 0.3% [0.0, 0.7] |

The control is never trained on. After fine-tuning it should stay near 1.7%; a clear rise means
the model forgot general English.

## Speed: offline single-stream RTF

RTF = processing time / audio duration, end to end (feature extraction + generate + decode), one
request at a time, 5 warm-up requests discarded, 20 clips per bucket × 3 repeats.

| Bucket | Requests | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| **Overall** | 141 | 0.111 | 0.107 | **0.164** | 0.213 |
| 2–5 s | 60 | 0.127 | 0.126 | 0.181 | 0.216 |
| 5–15 s | 60 | 0.103 | 0.103 | 0.139 | 0.153 |
| 15–30 s | 21 | 0.092 | 0.090 | 0.128 | 0.129 |

Throughput is 9.8 s of audio per wall-clock second, with peak GPU memory of 4.0 GB. The 15–30 s
bucket has only 7 eval clips, so its 21 requests are those 7 clips × 3 repeats.

## Observations

- **Malay conversational is the weakest slice (30.6% WER)**, three times worse than read Malay
  (10.5%). The worst clips are short, casual utterances that come out as phonetically similar
  nonsense (`ada orang kacau lah pokok nipah tu` → `Aduna kacawa, bukod diba tungo`). This is
  the main target for fine-tuning.
- **Language ID drifts on short Malay audio.** Of the 31 Malay conversational clips, one was
  detected as Tagalog and one as Indonesian. 14 of 62 Manglish clips were detected as English.
- **Short clips are hardest:** 23.2% WER on 2–5 s against 11.7% on 15–30 s, since there is less
  context to settle the language and vocabulary.
- **Some WER is formatting, not recognition.** The model writes numbers as words
  (`100,000` → `seratus ribu`, `2.0` → `dua puluh bulan kosong`) and the normalizer does not
  unify them. The Manglish references are Whisper pseudo-labels, so they also carry their own
  errors (for example, dropped words at clip edges).
- **The baseline is already faster than real time** (P95 RTF 0.16) with eager attention and
  batch 1, leaving SDPA, batching and vLLM as measured steps for the optimization work.

## Reproduce

```bash
uv sync --locked --group dev --extra train --extra data
uv run asr-assess data
uv run pytest -m model
uv run asr-assess eval --engine configs/engines/hf_base.yaml --manifest eval      # results/eval/hf_base-eval/
uv run asr-assess eval --engine configs/engines/hf_base.yaml --manifest control   # results/eval/hf_base-control/
uv run asr-assess bench --engine configs/engines/hf_base.yaml                     # results/bench/hf_base-offline/
```

Raw outputs: `metrics.json` and `hypotheses.jsonl` (per-clip reference, hypothesis, detected
language, error counts) under `results/eval/`, and `summary.json` and `timings.jsonl` under
`results/bench/hf_base-offline/`.