#!/usr/bin/env bash
# Start stock `vllm serve` from a configs/vllm/*.yaml file:
#   bash scripts/vllm_serve.sh configs/vllm/default.yaml
set -euo pipefail
mapfile -t args < <(uv run asr-assess vllm-args "${1:?usage: vllm_serve.sh configs/vllm/<step>.yaml}")
exec uv run --extra serve vllm "${args[@]}"
