#!/usr/bin/env bash
# Start two `vllm serve` instances, one after the other, and keep both running until Ctrl-C:
#   bash scripts/vllm_serve_two.sh                      # both on GPU 0 (two_instances_{a,b}.yaml)
#   GPU_B=1 bash scripts/vllm_serve_two.sh A.yaml B.yaml  # instance B on GPU 1
# B starts only after A is healthy, so A's memory profiling sees an otherwise empty GPU.
# Logs: logs/<config name>.server.log. Load-test both with
#   make loadtest URL=http://localhost:8000,http://localhost:8001 ...
set -euo pipefail
cd "$(dirname "$0")/.."
CFG_A=${1:-configs/vllm/two_instances_a.yaml}
CFG_B=${2:-configs/vllm/two_instances_b.yaml}
mkdir -p logs
pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait' EXIT INT TERM

start() {  # start <config> <gpu>
    local log="logs/$(basename "$1" .yaml).server.log"
    CUDA_VISIBLE_DEVICES=$2 bash scripts/vllm_serve.sh "$1" >"$log" 2>&1 &
    pids+=($!)
    echo "started $1 on GPU $2 (pid $!), log $log"
}

wait_healthy() {  # wait_healthy <config>
    local port
    port=$(uv run asr-assess vllm-args "$1" | sed -n '/^--port$/{n;p;}')
    until curl -sf "localhost:$port/health" >/dev/null; do
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "a server exited; see logs/"; exit 1; }
        done
        sleep 5
    done
    echo "$1 is up on port $port"
}

start "$CFG_A" "${GPU_A:-0}"
wait_healthy "$CFG_A"
start "$CFG_B" "${GPU_B:-0}"
wait_healthy "$CFG_B"
echo "both up; Ctrl-C stops both"
wait
