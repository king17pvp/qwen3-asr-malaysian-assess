# Thin aliases for `uv run asr-assess ...`; stage targets are added with their stages.
.PHONY: help install check-configs data data-plan test lint format

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

install: ## Base install + dev tools, and git hooks
	uv sync --group dev
	uv run pre-commit install

check-configs: ## Validate the shipped YAML configs
	uv run asr-assess check-config data configs/data.yaml
	uv run asr-assess check-config lora configs/lora.yaml
	uv run asr-assess check-config loadtest configs/loadtest.yaml

data-plan: ## Plan the dataset from Hub metadata only (no audio download)
	uv run --extra data asr-assess data --dry-run

data: ## Build data/manifests/{train,eval,control}.jsonl and the 16 kHz WAVs
	uv run --extra data asr-assess data

test: ## CPU-only test suite
	uv run pytest

lint: ## ruff + mypy, as in CI
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .
