#!/usr/bin/env bash
# One-time setup on a fresh Vast.ai CUDA box.
# Usage: HF_TOKEN=... bash scripts/vast_setup.sh <git-url>
set -euo pipefail

REPO_URL="${1:?usage: HF_TOKEN=... bash scripts/vast_setup.sh <git-url>}"
: "${HF_TOKEN:?export HF_TOKEN first (Hub downloads are rate-limited without it)}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d qwen3-asr-malaysian-assess ] || git clone "$REPO_URL" qwen3-asr-malaysian-assess
cd qwen3-asr-malaysian-assess
uv sync --locked --group dev --extra train --extra data
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; \
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
