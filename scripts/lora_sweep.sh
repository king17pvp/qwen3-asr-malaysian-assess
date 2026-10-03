#!/usr/bin/env bash
# Train, merge and evaluate (eval + control) every LoRA sweep config, one after another:
#   bash scripts/lora_sweep.sh                                   # all configs/lora_rank*_dropout10_*.yaml
#   bash scripts/lora_sweep.sh configs/lora_rank8_dropout10_lr5e-5.yaml   # just these
# A step is skipped when its output already exists (so a crashed sweep resumes); FORCE=1 reruns
# everything. A failed config is logged and the sweep moves on to the next one.
# Per-step outputs:
#   results/train/<config>/train_summary.json
#   results/merge/<config>/merge_summary.json
#   results/eval/hf_ft_<suffix>-{eval,control}/metrics.json
# plus a WER/CER roll-up of all configs in results/sweep/summary.json.
set -uo pipefail

cd "$(dirname "$0")/.."
FORCE="${FORCE:-0}"
LOG_DIR="logs/sweep"
mkdir -p "$LOG_DIR" results/sweep

if [[ $# -gt 0 ]]; then
    configs=("$@")
else
    configs=(configs/lora_rank*_dropout10_*.yaml)
fi

done_or_force() { [[ "$FORCE" != 1 && -f "$1" ]]; }

run_step() {  # run_step <label> <log> <cmd...>
    local label=$1 log=$2
    shift 2
    echo "[$(date '+%F %T')] $label -> $log"
    if ! "$@" >"$log" 2>&1; then
        echo "[$(date '+%F %T')] FAILED: $label (tail of $log below)"
        tail -n 20 "$log"
        return 1
    fi
}

run_config() {
    local config=$1
    local name engine
    name=$(basename "$config" .yaml)            # lora_rank16_dropout10_lr1e-4
    engine="configs/engines/hf_ft_${name#lora_}.yaml"  # hf_ft_rank16_dropout10_lr1e-4.yaml
    [[ -f "$engine" ]] || { echo "Missing engine config $engine"; return 1; }

    echo "=== $name ==="
    if done_or_force "results/train/$name/train_summary.json"; then
        echo "train: already done, skipping"
    else
        run_step "train $name" "$LOG_DIR/$name-train.log" \
            uv run --extra train asr-assess train --config "$config" || return 1
    fi

    if done_or_force "results/merge/$name/merge_summary.json"; then
        echo "merge: already done, skipping"
    else
        run_step "merge $name" "$LOG_DIR/$name-merge.log" \
            uv run --extra train asr-assess merge --config "$config" --engine "$engine" || return 1
    fi

    local manifest
    for manifest in eval control; do
        if done_or_force "results/eval/$(basename "$engine" .yaml)-$manifest/metrics.json"; then
            echo "eval ($manifest): already done, skipping"
        else
            run_step "eval $name on $manifest" "$LOG_DIR/$name-eval-$manifest.log" \
                uv run --extra train asr-assess eval --engine "$engine" --manifest "$manifest" \
                || return 1
        fi
    done
}

failed=()
for config in "${configs[@]}"; do
    run_config "$config" || failed+=("$config")
done

# Roll up overall WER/CER (with bootstrap CIs) of every sweep config that has eval results.
python3 - "${configs[@]}" <<'EOF'
import json, sys
from pathlib import Path

summary = {}
for config in sys.argv[1:]:
    name = Path(config).stem
    engine = "hf_ft_" + name.removeprefix("lora_")
    row = {}
    for manifest in ("eval", "control"):
        path = Path("results/eval") / f"{engine}-{manifest}" / "metrics.json"
        if path.exists():
            overall = json.loads(path.read_text())["report"]["overall"]
            row[manifest] = {"n": overall["n"], "wer": overall["wer"], "cer": overall["cer"]}
    summary[name] = row

out = Path("results/sweep/summary.json")
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"\n{'config':<34}{'eval WER':>10}{'eval CER':>10}{'ctrl WER':>10}{'ctrl CER':>10}")
for name, row in summary.items():
    cells = [f"{row[m][k]['value']:.4f}" if m in row else "-" for m in ("eval", "control") for k in ("wer", "cer")]
    print(f"{name:<34}" + "".join(f"{c:>10}" for c in cells))
print(f"\nWrote {out}")
EOF

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed configs: ${failed[*]}"
    exit 1
fi
echo "All ${#configs[@]} configs done."
