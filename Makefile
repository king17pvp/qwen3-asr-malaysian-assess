# Thin aliases for `uv run asr-assess ...`; stage targets are added with their stages.
.PHONY: help install check-configs data data-plan eval-base bench-base train-smoke train merge push eval-ft serve-hf serve-vllm serve-vllm-two loadtest report test lint format

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

install: ## Base install + dev tools, and git hooks
	uv sync --group dev
	uv run pre-commit install

check-configs: ## Validate the shipped YAML configs
	uv run asr-assess check-config data configs/data.yaml
	uv run asr-assess check-config lora configs/lora.yaml
	uv run asr-assess check-config loadtest configs/loadtest.yaml
	uv run asr-assess check-config engine configs/engines/hf_base.yaml
	uv run asr-assess check-config engine configs/engines/hf_ft.yaml
	uv run asr-assess check-config eval configs/eval.yaml
	uv run asr-assess check-config bench configs/bench.yaml
	uv run asr-assess check-config engine configs/engines/vllm_ft.yaml
	uv run asr-assess check-config vllm configs/vllm/default.yaml
	uv run asr-assess check-config vllm configs/vllm/tuned.yaml
	uv run asr-assess check-config vllm configs/vllm/cpu_path.yaml
	uv run asr-assess check-config vllm configs/vllm/fp8.yaml
	uv run asr-assess check-config vllm configs/vllm/eager.yaml
	uv run asr-assess check-config vllm configs/vllm/fp8-kv.yaml
	uv run asr-assess check-config vllm configs/vllm/fp8-w.yaml
	uv run asr-assess check-config vllm configs/vllm/two_instances_a.yaml
	uv run asr-assess check-config vllm configs/vllm/two_instances_b.yaml
	uv run asr-assess check-config vllm configs/vllm/dp2.yaml
	uv run asr-assess check-config engine configs/engines/hf_ft_sdpa.yaml
	uv run asr-assess check-config serve configs/serve/hf_baseline.yaml
	uv run asr-assess check-config serve configs/serve/hf_sdpa.yaml
	uv run asr-assess check-config serve configs/serve/hf_batched.yaml

data-plan: ## Plan the dataset from Hub metadata only (no audio download)
	uv run --extra data asr-assess data --dry-run

data: ## Build data/manifests/{train,eval,control}.jsonl and the 16 kHz WAVs
	uv run --extra data asr-assess data

eval-base: ## Baseline WER/CER on eval and control (GPU box, train extra)
	uv run --extra train asr-assess eval --engine configs/engines/hf_base.yaml --manifest eval
	uv run --extra train asr-assess eval --engine configs/engines/hf_base.yaml --manifest control

bench-base: ## Baseline single-stream RTF per bucket (GPU box, train extra)
	uv run --extra train asr-assess bench --engine configs/engines/hf_base.yaml

train-smoke: ## Two LoRA steps on a few clips: run first on a new GPU box (train extra)
	uv run --extra train asr-assess train --smoke

train: ## Decoder-only LoRA fine-tuning; best epoch by dev loss (GPU box, train extra)
	uv run --extra train asr-assess train

merge: ## Merge the best adapter, verify weight deltas, reload + transcribe (train extra)
	uv run --extra train asr-assess merge --engine configs/engines/hf_ft.yaml

push: ## Upload the verified merge + model card to the Hub as REPO (private; data extra)
	uv run --extra data asr-assess push --repo $(REPO)

eval-ft: ## Fine-tuned WER/CER on eval and control, same settings as eval-base
	uv run --extra train asr-assess eval --engine configs/engines/hf_ft.yaml --manifest eval
	uv run --extra train asr-assess eval --engine configs/engines/hf_ft.yaml --manifest control

serve-hf: ## HF server from CFG (default configs/serve/hf_baseline.yaml; train + http extras)
	uv run --extra train --extra http asr-assess serve --config $(or $(CFG),configs/serve/hf_baseline.yaml)

serve-vllm: ## Stock vllm serve from CFG (default configs/vllm/default.yaml; serve extra)
	bash scripts/vllm_serve.sh $(or $(CFG),configs/vllm/default.yaml)

serve-vllm-two: ## Two vllm serve instances (default: both on GPU 0, ports 8000/8001; serve extra)
	bash scripts/vllm_serve_two.sh

loadtest: ## Load test URL as LABEL; PROFILE=quick|full, SERVER_CFG copied into the summary
	uv run --extra http asr-assess loadtest --url $(URL) --label $(LABEL) --profile $(or $(PROFILE),quick) $(if $(SERVER_CFG),--server-config $(SERVER_CFG))

report: ## Tables and plots in results/plots from results/loadtest (report extra)
	uv run --extra report asr-assess report

test: ## CPU-only test suite
	uv run pytest

lint: ## ruff + mypy, as in CI
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .
