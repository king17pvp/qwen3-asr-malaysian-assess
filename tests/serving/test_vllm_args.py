"""Tests for turning a vLLM YAML config into `vllm serve` arguments."""

from asr_assess.core.config import VLLMServeConfig
from asr_assess.serving.vllm_args import vllm_args


def test_minimal_config_is_stock_vllm() -> None:
    assert vllm_args(VLLMServeConfig(model="m", port=8000)) == ["serve", "m", "--port", "8000"]


def test_set_fields_become_flags_and_bools_are_bare() -> None:
    cfg = VLLMServeConfig(
        model="m",
        port=8000,
        max_num_seqs=64,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        extra_args=["--x", "1"],
    )
    args = vllm_args(cfg)
    assert args[:4] == ["serve", "m", "--port", "8000"]
    assert "--enforce-eager" in args
    assert args[args.index("--max-num-seqs") + 1] == "64"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[-2:] == ["--x", "1"]


def test_false_bools_are_omitted() -> None:
    assert "--enforce-eager" not in vllm_args(VLLMServeConfig(model="m", port=1))
