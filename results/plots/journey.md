| Configuration | Profile | Max @0.5 | Max @0.3 | P95 RTF at max | Peak VRAM GiB | WER at max |
|---|---|---|---|---|---|---|
| smoke-vllm | quick | 88 | 64 | 0.337 | 22.3 | 14.4% |
| vllm-default | full | 112 | 112 | 0.261 | 23.4 | 14.7% |
| vllm-tuned-q1 | quick | 104 | 104 | 0.191 | 19.5 | 14.8% |
| vllm-default-u | full | 116 | 64 | 0.438 | 24.0 | 14.6% |
| vllm-tuned-u-q1 | quick | 128 | 64 | 0.464 | 20.3 | 14.2% |
| vllm-cpu-path-u | quick | 128 | 64 | 0.460 | 21.5 | 14.3% |
| vllm-cpu-path-u-1 | quick | 128 | 64 | 0.456 | 21.5 | 14.2% |
| vllm-eager-u | quick | 128 | 64 | 0.463 | 23.9 | 14.3% |
| vllm-eager-u-1 | quick | 128 | 64 | 0.457 | 20.3 | 14.2% |
| hf-baseline | full | 1 | 1 | 0.222 | 5.4 | 14.4% |
| hf-sdpa | quick | 1 | 1 | 0.184 | 5.4 | 14.6% |
| hf-batched-b8-w10 | quick | 1 | 1 | 0.187 | 5.4 | 14.1% |
| hf-batched-b8-w30 | quick | 1 | 1 | 0.202 | 5.4 | 15.5% |
| hf-batched-b16-w10 | quick | 1 | 1 | 0.180 | 5.4 | 14.3% |
| hf-batched-b16-w30 | quick | 1 | 1 | 0.189 | 5.4 | 15.0% |
| hf-batched-b32-w10 | quick | 1 | 1 | 0.183 | 5.4 | 14.0% |
| hf-batched-b32-w30 | quick | 1 | 1 | 0.203 | 5.4 | 15.1% |
| hf-batched-b16-w10-probe | quick | 1 | 1 | 0.184 | 5.4 | 14.3% |
| final | full | 116 | 64 | 0.479 | 23.1 | 14.6% |
| vllm-fp8-marlin-u | full | 104 | 64 | 0.383 | 23.4 | 18.3% |
| vllm-fp8kv-u | full | 104 | 64 | 0.387 | 24.0 | 14.6% |
