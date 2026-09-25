# Thin aliases for `uv run asr-assess ...`; stage targets are added with their stages.
.PHONY: help install check-configs data data-plan eval-base bench-base test lint format

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
	uv run asr-assess check-config eval configs/eval.yaml
	uv run asr-assess check-config bench configs/bench.yaml

data-plan: ## Plan the dataset from Hub metadata only (no audio download)
	uv run --extra data asr-assess data --dry-run

data: ## Build data/manifests/{train,eval,control}.jsonl and the 16 kHz WAVs
	uv run --extra data asr-assess data

eval-base: ## Baseline WER/CER on eval and control (GPU box, train extra)
	uv run --extra train asr-assess eval --engine configs/engines/hf_base.yaml --manifest eval
	uv run --extra train asr-assess eval --engine configs/engines/hf_base.yaml --manifest control

bench-base: ## Baseline single-stream RTF per bucket (GPU box, train extra)
	uv run --extra train asr-assess bench --engine configs/engines/hf_base.yaml

test: ## CPU-only test suite
	uv run pytest

lint: ## ruff + mypy, as in CI
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .
