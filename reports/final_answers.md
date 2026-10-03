# Final answers

Details and evidence: `reports/inference.md` (serving) and `reports/finetuning.md` (model). All
numbers are from one RTX 3090 (24 GB) with an Intel i7-8700, serving the fine-tuned
`king17pvp/qwen3-asr-1.7b-malaysian` with vLLM 0.30.0 in bf16 (`configs/vllm/tuned.yaml`), unique
audio per request, full load-test profile.

## 1. Maximum concurrency at P95 RTF ≤ 0.5

**116 concurrent streams** (P95 RTF 0.479, 0 failed requests, WER 14.6%), and **64 streams at the
stronger P95 ≤ 0.3** (P95 0.284).

| | Streams | P95 RTF | Throughput | GPU util | WER |
|---|---:|---:|---:|---:|---:|
| P95 ≤ 0.5 | **116** | 0.479 | 440 audio-s/s (73.6 req/s) | 85% | 14.6% |
| P95 ≤ 0.3 | **64** | 0.284 | 391 audio-s/s (65.3 req/s) | 79% | 14.7% |
| For reference: Transformers baseline | 1 | 0.222 | 7.5 audio-s/s | 34% | 14.4% |

A stream is a closed-loop client: it sends the next utterance as soon as the previous transcript
returns, so each stream sends audio several times faster than real time. 120 and 128 streams fail
(P95 0.525 / 0.514, plus 3 dropped connections each). WER stays at 14.5–14.9% at every level, and
offline WER through the same server is 14.71% vs 14.77% through Transformers: serving costs no
accuracy.

## 2. Main bottleneck

**One CPU core: the single-threaded vLLM EngineCore process.** From 64 streams up it runs at
98–100% of one core while the GPU is at 75–88% utilization, the KV cache is at most 17% full and no
request ever waits in the scheduler.

The ceiling is about **78 requests per second** (~470 audio-s/s), and every vLLM configuration hits
the same one: defaults, tuned scheduler settings, two API-server processes, no CUDA graphs, FP8 KV
cache and FP8 weights. Changes that cut GPU work or per-step CPU work do not move it, so the cost is
per request: about **12.7 ms of EngineCore CPU per request**, most likely handling each request's
audio features. Supporting evidence: the same Transformers benchmark ran 23% faster on a Ryzen 7
5800X box than on this i7-8700 with the same GPU model.

Limitation: the per-request attribution is inferred from these comparisons; a profiler could not
attach inside the rented container (no `ptrace`).

## 3. Largest improvement

**Continuous batching, by moving from Transformers to vLLM:** 1 → 116 streams, and 7.5 → 465
audio-s/s throughput (62×), at the same WER.

The Transformers path cannot get past 1 stream even with SDPA and dynamic batching, because of
head-of-line blocking: a request-level batch runs to completion, so a request that arrives during a
batch waits for the whole batch. P95 doubles from 1 to 2 streams whatever the batch size or wait
time (best of 6 settings: 0.18 → 0.53). vLLM admits new requests into the running batch at every
decode step and pages the KV cache, which removes both the wait and the padding waste.

Smaller steps, for scale: SDPA −17% single-stream P95; dynamic batching 2× throughput at 4 streams
(44.6 audio-s/s at most); CUDA graphs 2.6× lower latency at 1 stream in vLLM.

## 4. With one more week

In order of expected value:

1. **Two vLLM instances per GPU** behind a least-outstanding-requests balancer (each at ~0.45 of GPU
   memory; the model's weights are 3.8 GiB). This gives two EngineCores on two cores and uses the
   12–25% of GPU that sits idle at the ceiling. It is the most direct test of the bottleneck and the
   most likely capacity gain.
2. **Profile EngineCore** (`py-spy` on a host that allows it) during the 128-stream level to find
   the ~12.7 ms per request, then remove it: e.g. a more compact audio-feature path, or moving
   feature work out of the engine loop. Possibly a vLLM upstream fix.
3. **Benchmark on a CPU with faster single cores** (and record CPU model in every comparison):
   on this hardware, capacity is set by single-thread speed.
4. **An open-loop load test** with utterances arriving at real-time speed per session (with pauses
   between utterances), which is what real sessions look like and what question 5 needs. The
   closed-loop test overstates load per stream.
5. **Fix the two measurement gaps:** read vLLM's `finish_reason` so runaway generations show up in
   the eval (the FP8 loop was invisible to it), and find the rare `ReadError` connection drops at
   ≥ 108 streams.
6. **Accuracy:** more Malay conversational data (the weakest slice at 27% WER, trained on 5 minutes
   from 6 speakers), and a paired bootstrap for the before/after comparison.
7. **Quantization done properly:** online FP8 weights matched bf16 on 122 of 123 clips but sent
   one clip into a deterministic repetition loop (eval WER 18.78% from that clip alone). A calibrated offline checkpoint (INT8 W8A8, native on Ampere) plus a per-clip loop check
   and a length cap tied to audio duration. Low priority while the CPU is the limit.

## 5. Architecture for 1,000 simultaneous sessions

**Sizing.** Two ways to read "session", giving a range:

- **As closed-loop streams** (the load test's definition, a worst case: each session sends
  continuously, faster than real time): 1000 / 116 ≈ 9 GPUs at the limit. With headroom (run at
  ~80 streams per GPU, between the measured 64 at P95 0.28 and 96 at 0.38): 1000 / 80 ≈ 13, plus one
  spare: **about 14 GPUs**.
- **As live audio** (each session produces at most 1 second of speech per second): 1000 sessions
  need at most ~1000 audio-s/s. One GPU sustains ~440 audio-s/s at P95 ≤ 0.5; at 75% of that
  (~330 audio-s/s) it is **about 4 GPUs, plus one spare = 5**. Real sessions also pause between
  utterances, so this is an upper bound for that reading.

I would plan on **5–6 GPU replicas for live sessions**, validate it with the open-loop test from
question 4 before committing, and keep the ~14-GPU figure as the bound for batch-style clients that
submit audio back-to-back. Two instances per GPU or faster CPUs would lower both numbers.

**Design:**

```
clients ──WebSocket/gRPC──▶ ingest gateway (stateless, autoscaled on CPU)
                              • session auth, rate limits
                              • VAD: cut the stream into utterances ≤ 30 s
                              • audio decode/resample to 16 kHz once
                                     │  one HTTP request per utterance
                                     ▼
                        load balancer: least outstanding requests
                                     │
              ┌──────────────┬───────┴──────┬──────────────┐
              ▼              ▼              ▼              ▼
          vLLM pod 1     vLLM pod 2   …  vLLM pod N   (1 GPU each, 1–2 instances,
          bf16, tuned    …                             host with fast single cores)
```

- **Gateway separate from inference:** sessions are long-lived and cheap; utterances are short and
  GPU-bound. The gateway keeps per-session state (partial transcripts, ordering) so any GPU can
  serve any utterance, and pods stay stateless.
- **Routing by in-flight requests, not round-robin:** utterance lengths vary 2–30 s, so outstanding
  work per pod is the right signal. Cap in-flight requests per pod near the measured limit (~100) so
  overload queues at the balancer instead of raising everyone's latency.
- **Autoscaling on in-flight requests per pod and P95 RTF** (vLLM `/metrics`: running, waiting;
  the gateway's own latency histogram), not on GPU utilization, which never reaches 100% here.
  Keep warm spares: a vLLM pod takes ~2 minutes to start (weights + CUDA graph capture).
- **Hosts chosen for CPU single-thread speed** as well as the GPU; budget at least 2 fast cores
  per engine instance plus cores for the API server.
- **Reliability:** retry an utterance once on a dropped connection (requests are idempotent); health
  checks on `/health`; N+1 replicas; per-pod dashboards for P95 RTF, req/s, EngineCore CPU and GPU
  utilization.
- **Quality guard:** cap `max_tokens` relative to utterance duration and flag `finish_reason =
  length`, so a runaway generation cannot hold a slot or return garbage silently.
