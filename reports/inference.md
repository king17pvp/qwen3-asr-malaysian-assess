# Inference optimization: Qwen3-ASR 1.7B (fine-tuned) on one RTX 3090

**Headline.** The final configuration (vLLM 0.30.0, `configs/vllm/tuned.yaml`, bf16) sustains
**116 concurrent streams at P95 RTF ≤ 0.5** (P95 0.479) and **64 streams at P95 ≤ 0.3** (P95 0.284),
with no failed requests and WER flat at 14.5–14.9% at every level. The plain Transformers baseline
sustains **1** stream. Throughput goes from 7.5 to 449 audio-seconds per second (60×).

The largest gain comes from one change: **continuous batching** (vLLM), which removes the
head-of-line blocking of request-level batching. The final system is limited by **one CPU core**: the
single-threaded vLLM EngineCore process runs at ~100% of a core from 64 streams up, while the GPU sits
at 75–88% utilization and the KV cache never passes 17%.

All numbers below come from `results/` (raw JSON/JSONL per run) and the tables and plots
`make report` writes to `results/plots/`. Nothing is typed in by hand.

## 1. Setup

### Environment

All load tests and the Part 2 baseline ran on the same machine.

| | |
|---|---|
| Machine | Vast.ai instance `C.53713864` |
| GPU | NVIDIA GeForce RTX 3090, 24 GB (sm86, Ampere), driver 560.35.03 |
| CPU / RAM | Intel Core i7-8700 @ 3.2 GHz (6 cores / 12 threads), 31.3 GiB |
| OS | Linux 5.15.0-139, Python 3.12.3 |
| vLLM server | vLLM 0.30.0 (cu129 wheel), torch 2.13.0+cu129, CUDA runtime 12.9 |
| HF server / offline bench | torch 2.14.0+cu126, transformers 5.17.0, CUDA runtime 12.6 |
| Load-test client | same machine, httpx 0.28.1, NVML sampling at 10 Hz, vLLM `/metrics` at 1 Hz |

Fine-tuning and the first vLLM accuracy check (`results/eval/vllm_ft-*`) ran on a different box
(`C.53477159`, Xeon E5-2696 v3, RTX 3090). The early `smoke-vllm` run also ran there and is not used
below.

### Model

`king17pvp/qwen3-asr-1.7b-malaysian`: `Qwen/Qwen3-ASR-1.7B-hf` with the rank-16 LoRA adapter merged
(see `reports/finetuning.md`). bf16 unless a row says otherwise. Greedy decoding,
`max_new_tokens` 256, no language hint.

### Definitions and protocol

- **Stream:** a closed-loop client. It sends one utterance, waits for the transcript, then sends the
  next one immediately.
- **RTF** = end-to-end request latency (including queueing) / audio duration.
- **Audio pool:** the 123 held-out eval clips (2–30 s, mixed buckets), sampled with a fixed seed.
- **Unique audio per request:** every request's WAV carries a random 64-bit tag in the lowest bit of
  its first 64 samples (`core/transcription_api.py: tag_wav`). Each sample moves by at most 1 LSB, so
  the audio is unchanged for the model, but vLLM can no longer reuse its prefix or multimodal caches
  across repeats of a clip. Real traffic never repeats audio (see §5.1 for why this matters).
- **Levels:** 1, 2, 4, 8, 16, 32, 64, 128 streams, plus bisection between the last passing level
  and the first failing one.
- **Profiles:** `full` = 30 s warm-up (discarded) + 3 × 120 s measured, 4 bisection steps. `quick` =
  15 s warm-up + 1 × 60 s, 3 bisection steps. Only `full` runs give headline numbers; `quick` runs
  are for comparing changes against each other.
- **Max sustainable** = the highest level with P95 RTF ≤ 0.5, no timeouts or errors, and WER within
  1 point of the run's single-stream WER. The same rule at 0.3 gives the stronger number.
- **VRAM** is device memory used as seen by NVML. For vLLM this is mostly the up-front reservation
  (`gpu_memory_utilization`), not what the requests need.

## 2. Baseline

### Part 2: offline, single stream (pretrained model)

`make bench-base`: pretrained `Qwen/Qwen3-ASR-1.7B-hf`, Transformers, eager attention, batch 1, up to
20 clips per bucket, 3 repeats (`results/bench/hf_base-offline/summary.json`).

| Bucket | Requests | Avg RTF | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| 2–5 s | 60 | 0.161 | 0.161 | 0.230 | 0.280 | 0.285 |
| 5–15 s | 60 | 0.133 | 0.133 | 0.177 | 0.197 | 0.199 |
| 15–30 s | 21 | 0.119 | 0.117 | 0.172 | 0.173 | 0.173 |
| **Overall** | 141 | 0.143 | 0.136 | **0.212** | 0.277 | 0.285 |

Throughput 7.6 audio-s/s, peak GPU memory 4.0 GB. GPU utilization was not recorded by the offline
bench; the same model architecture under the HF server at 1 stream used 34% (`hf-baseline`, below).

- Short clips have the worst RTF: fixed per-request costs (feature extraction, prompt prefill,
  `generate()` setup) are spread over less audio.
- The 15–30 s bucket has only 7 eval clips (21 requests with repeats), so its P95 is a thin estimate.
- The run's record has `git_dirty: true` at `76b525e` (untracked local configs; no tracked file
  changed).
- **The same benchmark on a faster CPU was 23% faster.** The first baseline
  (`reports/ResultsBeforeFineTuning.md`, 2026-09-26, commit `611456e`) ran the identical bench on
  another RTX 3090 box with an AMD Ryzen 7 5800X: P95 0.164 and 9.8 audio-s/s, against 0.212 and
  7.6 here on the i7-8700. Same GPU model, same model, same settings: batch-1 inference on this model
  is already sensitive to single-core CPU speed, which foreshadows the bottleneck in §4.

### Baseline under load (fine-tuned model)

`hf-baseline` (full): the HF server (`configs/serve/hf_baseline.yaml`), eager attention, one request
at a time.

| Streams | Avg RTF | P50 | P95 | GPU util % | VRAM GiB | Audio-s/s | Req/s | WER |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.147 | 0.140 | 0.222 | 34 | 5.4 | 7.5 | 1.27 | 14.4% |
| 2 | 0.336 | 0.293 | 0.659 | 34 | 5.4 | 7.5 | 1.27 | 14.0% |
| 4 | 0.714 | 0.664 | 1.458 | 34 | 5.4 | 7.6 | 1.28 | 14.1% |

Max sustainable: **1** at both thresholds. Throughput is flat at 7.5 audio-s/s because the server
handles one request at a time, so extra streams only queue and P95 roughly doubles with each doubling
of streams. The GPU is idle two thirds of the time (34% utilization) and uses 5.4 of 24 GiB: batch-1
decoding of a 1.7B model is bound by per-token kernel launches and Python overhead, not by GPU compute.

## 3. Optimization journey

| Configuration | Run | Profile | Max streams P95 ≤ 0.5 | P95 ≤ 0.3 | P95 RTF at max | Peak throughput (audio-s/s) | GPU util at max | Peak VRAM GiB | WER at max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (HF, eager, batch 1) | `hf-baseline` | full | 1 | 1 | 0.222 | 7.6 | 34% | 5.4 | 14.4% |
| FlashAttention/SDPA | `hf-sdpa` | quick | 1 | 1 | 0.184 | 9.1 | 38% | 5.4 | 14.6% |
| Dynamic batching (b16, 10 ms) | `hf-batched-b16-w10` | quick | 1 | 1 | 0.180 | 19.0 (44.6 in the probe) | 38% | 9.1 (23.0 in the probe) | 14.3% |
| vLLM (defaults) | `vllm-default-u` | full | 116 | 64 | 0.438 | 465.3 | 87% | 24.0 | 14.6% |
| vLLM tuned | `vllm-tuned-u-q1` | quick | ≥ 128 | 64 | 0.464 | 474.1 | 88% | 23.4 | 14.2% |
| CPU-path fixes (2 API servers) | `vllm-cpu-path-u-1` | quick | ≥ 128 | 64 | 0.456 | 474.0 | 88% | 21.5 | 14.2% |
| FP8 KV cache | `vllm-fp8kv-u` | full | 104 | 64 | 0.387 | 469.2 | 80% | 24.0 | 14.6% |
| FP8 weights + KV cache | `vllm-fp8-marlin-u` | full | 104* | 64* | 0.383 | 471.5 | 79% | 23.4 | **18.3%** |
| Eager ablation (no CUDA graphs) | `vllm-eager-u-1` | quick | ≥ 128 | 64 | 0.457 | 473.6 | 88% | 23.9 | 14.2% |
| **Final** (`vllm/tuned.yaml`) | `final` | **full** | **116** | **64** | **0.479** | 449.0 | 85% | 23.2 | 14.6% |

\* Passes the load-test rule because WER under load is compared with the same run's single-stream
WER (18.4%), but fails the offline accuracy check against bf16 (§3.7). The whole gap comes from one
clip that loops every time it is replayed; the other 122 clips are within 0.25 points of bf16.

"≥ 128" means the top tested level passed (`max_num_seqs` is 128 in `tuned.yaml`). The full-profile
`final` run of the same configuration gives 116 (§5.4), so quick-profile rows are compared with each
other, not with full rows.

Plots: `results/plots/p95_rtf.png` (P95 RTF vs streams per run) and `results/plots/throughput.png`.
Per-run concurrency tables: `results/plots/concurrency_<run>.md`.

### 3.1 FlashAttention/SDPA

- **Hypothesis:** eager attention wastes time in unfused kernels; SDPA's fused kernels make each
  decode step cheaper.
- **Change:** `attn_implementation: sdpa` (`configs/serve/hf_sdpa.yaml`), still batch 1.
- **Measurement:** 1-stream P95 0.222 → 0.184 (−17%), throughput 7.5 → 9.1 audio-s/s (+21%), GPU
  34% → 38%, same VRAM and WER.
- **Result:** a real but small gain. A batch-1 step is still mostly launch and Python overhead, so
  the GPU stays ~60% idle. Max streams stays at 1 (P95 at 2 streams 0.522, just over 0.5): serving one
  request at a time caps concurrency whatever the per-step speed.

### 3.2 Dynamic batching

- **Hypothesis:** grouping concurrent requests into one `generate()` call fills the idle GPU time and
  the unused memory.
- **Change:** the batcher in `serving/batcher.py` (max batch B, max wait T ms). Swept
  B ∈ {8, 16, 32} × T ∈ {10, 30} ms, quick profile; best point b16-w10, now set in
  `configs/serve/hf_batched.yaml`.

  | B \ T | 10 ms | 30 ms |
  |---|---|---|
  | 8 | 1 / 0.544 / 18.7 | 1 / 0.584 / 18.9 |
  | 16 | **1 / 0.533 / 19.0** | 1 / 0.527 / 18.7 |
  | 32 | 1 / 0.523 / 18.7 | 1 / 0.594 / 18.0 |

  Cell: max streams / P95 RTF at 2 streams / audio-s/s at 4 streams.
- **Measurement:** throughput at 4 streams doubles (9.1 → 19.0 audio-s/s) and VRAM grows 5.4 → 9.1 GiB
  as batches form, but P95 at 2 streams is 0.52–0.59 at every point, so max streams stays 1. A probe
  run that kept raising the load (`hf-batched-b16-w10-probe`, not a journey row) peaks at
  **44.6 audio-s/s** from 32 streams up, with GPU at 63–64% and VRAM at 23.0 of 24 GiB.
- **Result:** batching raises throughput, not concurrency, because of **head-of-line blocking**. A
  batch runs `generate()` to completion; a request that arrives during a batch waits for the whole
  batch, then runs its own, so its latency is up to ~2× a batch's service time. Batches are also padded
  to the longest audio and decode until the longest transcript ends, so short requests pay for long
  ones, and the HF KV cache is allocated per batch for the padded length (not paged), so memory caps
  the batch size. `max_batch` does not matter at ≤ 4 streams (a batch never exceeds 4); 30 ms waits
  are slightly worse than 10 ms at 1 stream (added wait, no batch to gain).

### 3.3 vLLM (continuous batching)

- **Hypothesis:** iteration-level (continuous) batching removes the head-of-line wait: a new request
  joins the running batch at the next decode step. Paged KV memory removes the padding waste, and CUDA
  graphs cut the per-step launch overhead.
- **Change:** stock `vllm serve` (`configs/vllm/default.yaml`), OpenAI-compatible
  `/v1/audio/transcriptions`.
- **Measurement (`vllm-default-u`, full):** 1-stream P95 0.052 (4.3× lower than the HF baseline);
  **116 streams** at P95 ≤ 0.5 (P95 0.438) and 64 at ≤ 0.3 (0.278); throughput 465 audio-s/s at 128
  streams; WER 14.5–14.9% at every level. 120 and 128 failed on 1 and 2 `ReadError`s (connection
  dropped by the server, ~1 in 27k requests) while P95 was 0.484.
- **Result:** the largest step of the journey: 1 → 116 streams and 7.6 → 465 audio-s/s. Accuracy
  is unchanged: offline WER through vLLM is 14.71% vs 14.77% through Transformers on the same 123
  clips (`results/eval/vllm_ft-eval`, `results/eval/hf_ft_rank16_dropout10_lr1e-4-eval`); control
  set 1.70% in both.

### 3.4 vLLM tuned

- **Hypothesis:** the defaults cap concurrency or the prefill/encoder budget per step below what
  24 GB allows.
- **Change:** `configs/vllm/tuned.yaml`: `gpu_memory_utilization 0.92`, `max_num_seqs 128`,
  `max_num_batched_tokens 16384`, `max_model_len 4096`.
- **Measurement (`vllm-tuned-u-q1`, quick):** identical to the defaults within noise: P95 within
  ±0.002 of `vllm-default-u` from 1 to 64 streams, peak 474 vs 465 audio-s/s, GPU 76–88% in both,
  `waiting` 0 at every level, KV cache ≤ 15%.
- **Result:** no gain once caching is removed. Nothing queues in the scheduler and the KV cache is
  mostly empty, so the knobs that control them have nothing to fix. (With repeated audio the same
  change did give +60% peak throughput; that result was a caching artefact, §5.1.)

### 3.5 CPU-path fixes

- **Hypothesis:** with the GPU at ~88% while P95 grows, the API server process (multipart parsing,
  audio decoding, feature extraction) is saturating one core.
- **Change:** `api_server_count: 2` (`configs/vllm/cpu_path.yaml`).
- **Measurement (`vllm-cpu-path-u-1`, quick; repeat `vllm-cpu-path-u` agrees):** identical to tuned
  (474.0 vs 474.1 audio-s/s at 128 streams, P95 within ±0.008). `top -H` at 128 streams:
  `VLLM::EngineCore` at **98.3%** of one core; each API-server thread 10–18%; load-test client 16%;
  the other 11 cores 80–99% idle.
- **Result:** no gain, and the hypothesis is wrong: the API server is not busy. The saturated
  process is the **EngineCore** (scheduling, input preparation and output processing for every step,
  in one Python thread), which a second API server cannot help. This run located the bottleneck (§4).

### 3.6 Eager ablation

- **Hypothesis:** CUDA graphs matter most at low load (launch-bound steps); if the engine loop is
  CPU-bound, disabling them should hurt at high load too, since each step then costs more CPU.
- **Change:** `enforce_eager: true` (`configs/vllm/eager.yaml`).
- **Measurement (`vllm-eager-u-1`, quick; repeat `vllm-eager-u` agrees):** 1-stream P95 0.133 vs
  0.052 (CUDA graphs are worth **2.6×** at low load; GPU 35% vs 88%); the gap shrinks with load (64
  streams 0.294 vs 0.279) and closes at 128 streams: 473.6 vs 474.1 audio-s/s, 78.5 vs 78.6 req/s.
  EngineCore at 98.7% of one core.
- **Result:** CUDA graphs are kept (they are what makes low-load latency good), but the second half of
  the hypothesis is wrong: a change that adds a lot of CPU work *per step* leaves the ceiling where it
  was. The ceiling therefore scales with the number of *requests*, not steps (§4).

### 3.7 FP8

- **Hypothesis:** FP8 halves KV-cache bytes per token (more sequences fit) and weight bytes (less
  memory traffic per decode step). On Ampere there are no FP8 tensor cores, so FP8 weights run as
  weight-only W8A16 (Marlin kernel): no faster matmuls, only less data to read. Expected effect on
  capacity: none, since the KV cache never passed 17% and the GPU is not the limit.
- **Change:** `configs/vllm/fp8.yaml` (tuned + `quantization: fp8` + `kv_cache_dtype: fp8`), then the
  two halves separately: `fp8-w` (weights only) and `fp8-kv` (KV cache only).
- **Getting it to start:** vLLM 0.30.0 does not start with FP8 weights on this GPU. It picks
  `CutlassFP8ScaledMMLinearKernel`, whose support check only tests for CUDA, not for the sm89+ it
  needs, and it is listed before Marlin, so it wins on sm86 and crashes (first as an Inductor error,
  then, with Inductor skipped, in `cutlass_scaled_mm_sm80_epilogue`). `--linear-backend marlin`
  (now in `fp8.yaml`) selects `MarlinFP8ScaledMMLinearKernel` and the server starts. FP8 KV alone
  starts without help.
- **Measurement, accuracy** (offline eval through each server; bf16 vLLM on the same box: 14.71% /
  1.70%):

  | Variant | Eval WER (123 clips) | Control WER (44 clips) | Looping clip |
  |---|---:|---:|---|
  | FP8 weights + KV | **18.78%** | 1.70% | loops |
  | FP8 weights only | — | — | loops (188 words vs ~40) |
  | FP8 KV only | **14.53%** | 1.70% | clean (36 words) |

  Almost the whole FP8-weights loss is **one clip** (`manglish/prepared-pseudolabel-chunks_22301-1.wav`,
  a parliament roll call). Without it, bf16 is 14.42% and FP8 weights + KV 14.67% over the other 122
  clips. On that clip, every run with FP8 weights replaces the honorific "yang Boleh **Hormat**" with
  "yang Boleh **Mahadatuk**", and from that wrong token the greedy decode invents a list of ministers
  ("Datuk Seri Heng Swee Keat, Timbalan Menteri Kewangan …") until it hits the 256-token limit. It
  is deterministic (identical output on 3 separate requests).
- **Measurement, load (both full):** both reach **104** streams at P95 ≤ 0.5 and 64 at ≤ 0.3. 108 and
  112 fail on **one `ReadError` each** out of ~28k requests while P95 is under 0.5 (FP8 KV: 0.418,
  0.464), so 104 vs `final`'s 116 is the same stray-error effect as in `vllm-default-u`, not a real
  loss. Peak 78.5–78.8 req/s, the same ceiling as every other vLLM run. KV usage halves (max 0.08 vs
  0.17); KV capacity 250,480 tokens (FP8 KV) and 275,824 (FP8 weights + KV) vs 127,232 (bf16).
  FP8 weights do speed up low load: 1-stream P95 0.041 vs 0.053 (−23%), 40.4 vs 33.3 audio-s/s
  (+21%); the gap is gone by 64 streams.
- **Result:**
  - **FP8 KV: no accuracy cost, no capacity gain.** It doubles a resource that was never used.
  - **FP8 weights: faster at low load, and accurate on 122 of 123 clips** (14.67% vs 14.42%), but
    they send one clip into a deterministic repetition loop, and that single clip lifts eval WER to
    18.78%. The load test shows 18.0–18.7% at every level for the same reason: it replays the
    123-clip pool, so the looping clip recurs. One loop in 123 clips is still disqualifying for
    serving (256 tokens of invented text), so this needs a calibrated checkpoint and a loop guard
    before it could ship.
  - The final configuration stays bf16.

## 4. Bottleneck analysis: the final system

Concurrency table for `final` (`results/plots/concurrency_final.md`; full profile; tested
1 → 128, then bisection 96, 112, 120, 116):

| Streams | Avg RTF | P50 | P95 | GPU util % | VRAM GiB | Audio-s/s | Req/s | WER |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.034 | 0.033 | 0.053 | 88 | 19.6 | 33.3 | 5.53 | 14.6% |
| 2 | 0.040 | 0.039 | 0.061 | 89 | 19.6 | 56.9 | 9.53 | 14.6% |
| 4 | 0.045 | 0.043 | 0.069 | 85 | 20.2 | 99.9 | 16.69 | 14.7% |
| 8 | 0.056 | 0.054 | 0.088 | 80 | 20.5 | 159.1 | 26.57 | 14.7% |
| 16 | 0.078 | 0.075 | 0.122 | 76 | 21.2 | 227.6 | 38.01 | 14.8% |
| 32 | 0.116 | 0.111 | 0.179 | 75 | 22.3 | 307.0 | 51.24 | 14.9% |
| 64 | 0.182 | 0.175 | **0.284** | 79 | 23.0 | 391.0 | 65.26 | 14.7% |
| 96 | 0.247 | 0.237 | 0.383 | 84 | 23.1 | 430.1 | 72.03 | 14.7% |
| 112 | 0.277 | 0.266 | 0.440 | 85 | 23.1 | 449.0 | 74.97 | 14.6% |
| **116** | 0.298 | 0.272 | **0.479** | 85 | 23.1 | 439.8 | 73.56 | 14.6% |
| 120 | 0.319 | 0.278 | 0.525 | 86 | 23.1 | 432.5 | 72.38 | 14.5% |
| 128 | 0.322 | 0.292 | 0.514 | 87 | 23.2 | 446.6 | 74.61 | 14.6% |

No errors at ≤ 116; 3 `ReadError`s each at 120 and 128 (which also fail on P95). Offline WER through
this server: 14.71% (`results/eval/final-eval`).

**The limit is the single-threaded vLLM EngineCore process on one CPU core.** Evidence:

1. **Not the GPU.** Utilization is 75–88% at every level and *falls* from 88% at 1 stream to 75% at
   32: the GPU waits for the engine between steps.
2. **Not the scheduler or memory.** `waiting` is 0 at every level of every vLLM run, `running` tracks
   the number of streams, and the KV cache peaks at 17%. Doubling KV capacity (FP8 KV) changes nothing.
3. **One core is saturated.** `top -H` shows `VLLM::EngineCore` at 98–100% of one core in every vLLM
   configuration checked (tuned + 2 API servers, eager, `final` already at 64 streams, FP8 KV), while
   the API servers use 10–33%, the client 9–21%, and the other 11 cores are mostly idle.
4. **The ceiling is per request, not per step.** Every vLLM configuration ends at the same throughput:
   ~78.5 req/s / ~470 audio-s/s at their peak (defaults 77.8, tuned 78.6, 2 API servers 78.6,
   eager 78.5, FP8 KV 78.5, FP8 weights 78.8; `final` peaks a little lower at 75.0 in its full run). Eager mode, which costs far more CPU per
   step, still reaches it. So the EngineCore spends about 1 / 78.5 ≈ **12.7 ms of CPU per request**, and
   a 3.2 GHz core can do about 78 of those per second. A likely candidate is the handling of each
   request's audio features (received from the API server over ZMQ, batched and copied to the GPU for
   the encoder), which would also explain §5.1: with a multimodal-cache hit the features are not sent
   again and the ceiling rose to 126–202 req/s.
5. **Latency grows because of it.** Past ~64 streams, requests arrive faster than the engine can
   admit and finish them, so per-request latency grows with the number of streams while throughput
   stays flat (430–449 audio-s/s from 96 up).

**Limitation:** the per-request attribution in point 4 is inferred from the configuration comparison.
A profiler (`py-spy`) could not attach to EngineCore because the container did not allow `ptrace`, so
which function inside the engine loop spends the 12.7 ms was not measured.

## 5. Failed experiments and caveats

### 5.1 The first vLLM benchmark was wrong (cache-hot audio)

The first vLLM runs replayed the same 123 clips, and the server log showed `Prefix cache hit rate:
90.1%, MM cache hit rate: 100.0%`: vLLM was serving the audio preprocessing and prefill of repeated
clips from cache. Real traffic never repeats audio. With unique audio per request (§1):

| | Repeated audio | Unique audio |
|---|---:|---:|
| Defaults, peak throughput (full) | 755 audio-s/s (`vllm-default`) | 465 (`vllm-default-u`) |
| Defaults, P95 at 96 streams | 0.219 | 0.377 |
| Defaults, max streams at P95 ≤ 0.3 | 112 | 64 |
| Tuned, peak throughput (quick) | 1211 (`vllm-tuned-q1`, GPU 100%) | 474 (`vllm-tuned-u-q1`, GPU 88%) |

The cache-hot runs overstated throughput by ~60% for the defaults and made the tuned engine settings
look like a +60% win (755 → 1211) that disappears with unique audio. They are kept as an ablation and
are not journey rows. The HF rows are unaffected (no cross-request cache).

### 5.2 Changes that did not help

| Change | Expected | Measured |
|---|---|---|
| vLLM tuned engine settings | more concurrency / bigger batches | identical to defaults (§3.4) |
| 2 API server processes | relieve a saturated front end | identical; the front end was not the bottleneck (§3.5) |
| Async scheduling | overlap engine CPU work with GPU steps | already on: vLLM 0.30.0 enables it by default (`async_scheduling: None` → on), so there was nothing to change |
| FP8 KV cache | more KV capacity | capacity doubled, not used (§3.7) |
| FP8 weights | lower latency | −23% P95 at 1 stream, but one clip loops deterministically, lifting eval WER to 18.78% (§3.7) |
| HF batching `max_batch` 32 / `max_wait_ms` 30 | bigger batches | no better than 16 / 10 (§3.2) |

### 5.3 Rare connection errors decide some verdicts

At ≥ 108 streams, a few requests per run fail with `httpx.ReadError` (the server closes the
connection while the client waits): 1–3 per ~27k requests, never at ≤ 104 streams, with no
traceback or OOM in the server log. Under the rule "no errors", one such failure fails a level even
when P95 is under 0.5 (`vllm-default-u` at 120/128, both FP8 runs at 108/112). It shifts the reported
maximum by a few streams, not the capacity picture. The cause (likely a keep-alive or socket limit
in the HTTP layer) was not investigated.

### 5.4 Quick vs full profile

The quick runs passed 128 streams with P95 0.456–0.464; the full run of the same configuration
(`final`) gives 0.514 at 128 and stops at 116. Longer warm-up and three 120 s windows catch more of
the tail. Quick runs are therefore used only to compare configurations against each other, and the
headline comes from `final` (full). Repeat runs agree closely: `vllm-cpu-path-u` vs `-u-1` and
`vllm-eager-u` vs `-u-1` differ by ≤ 0.007 in P95 and ≤ 1% in throughput at 128 streams.

### 5.5 WER under load

WER is flat for every vLLM run (14.2–15.1% across levels, within 1 point of single stream). The HF
batching probe shows 15.9% and 16.9% at 64 and 128 streams; at P95 RTF 4–8 most long clips do not
finish inside the measurement window, so completed requests lean towards short clips, which have
higher WER (for the pretrained model: 23.2% on 2–5 s clips vs 14.2% and 11.7% on longer ones,
`reports/ResultsBeforeFineTuning.md`). This was not
checked per bucket, so it is not claimed as a model effect; those levels fail P95 by 8–16× anyway.

### 5.6 Tooling gap: runaway generations are invisible on vLLM

The eval records `hit_token_limit` per clip, but the vLLM HTTP engine cannot see the finish reason
and always reports `False`, so the FP8 loop (which ran to the 256-token limit) showed 0 truncations.
It was found from the per-clip error diff. Reading vLLM's `finish_reason` would catch it.

## 6. Recommendations

1. **Run two vLLM instances on the GPU** (each `gpu_memory_utilization` ~0.45; the 1.7B model
   needs ~3.8 GiB of weights) behind a round-robin balancer. That gives two EngineCores on two cores
   for the same GPU, which is still 12–25% idle at the ceiling. This is the most direct test of §4 and
   the most likely next capacity gain. Not run for lack of time.
2. **Size CPUs for the engine, not just the GPU.** On this box, capacity is set by single-core speed:
   a CPU with faster cores should raise the ~78 req/s ceiling directly (the HF baseline was already
   23% faster on a Ryzen 7 5800X than on this i7-8700 with the same GPU, §2). When renting, prefer
   high single-thread performance over core count.
3. **Profile EngineCore** on a host that allows `ptrace` (`py-spy record` during the 128-stream level)
   to confirm where the ~12.7 ms per request goes, then look at reducing per-request feature handling
   (e.g. sending audio features in a more compact form, or moving that work out of the engine loop).
4. **Keep bf16 weights on Ampere.** If memory or latency ever requires quantization, use a calibrated
   offline checkpoint (e.g. INT8 W8A8, which Ampere runs natively) rather than online FP8, re-run the
   per-clip loop check, and add a guard against repetition (a length cap relative to audio duration).
5. **Fix the two measurement gaps:** read `finish_reason` in the vLLM engine (§5.6) and find the
   cause of the rare `ReadError`s (§5.3).
6. **Benchmark with unique audio.** Any vLLM benchmark that replays a fixed clip set needs the same
   tagging, or it measures the cache.

## 7. Reproduce

```bash
uv sync --extra serve                       # vLLM server env (conflicts with `train`)
make serve-vllm CFG=configs/vllm/tuned.yaml 2>&1 | tee logs/final.server.log
make loadtest URL=http://localhost:8000 LABEL=final PROFILE=full SERVER_CFG=configs/vllm/tuned.yaml
uv run --extra http asr-assess eval --engine configs/engines/vllm_ft.yaml --manifest eval --run-name final-eval

uv sync --extra train --extra http          # HF server env
make serve-hf CFG=configs/serve/hf_baseline.yaml
make loadtest URL=http://localhost:8001 LABEL=hf-baseline PROFILE=full SERVER_CFG=configs/serve/hf_baseline.yaml

make bench-base                             # Part 2 offline baseline (train env)
make report                                 # results/plots/: journey, concurrency tables, plots
```

Every result file records the git commit, config, GPU, package versions and a timestamp.
