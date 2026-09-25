"""Tests for the hardware/software snapshot parsers."""

import pytest

from asr_assess.benchmark.env_info import (
    EnvInfo,
    GpuInfo,
    collect_env_info,
    parse_cpu_model,
    parse_mem_total_gib,
    parse_nvidia_smi,
)


def test_parses_nvidia_smi_csv() -> None:
    text = "NVIDIA GeForce RTX 4090, 550.54.14, 24564\nNVIDIA GeForce RTX 4090, 550.54.14, 24564\n"
    assert parse_nvidia_smi(text) == [GpuInfo("NVIDIA GeForce RTX 4090", "550.54.14", 24564)] * 2


def test_no_nvidia_smi_output_means_no_gpus() -> None:
    assert parse_nvidia_smi("") == []


def test_parses_cpu_model() -> None:
    cpuinfo = "processor\t: 0\nmodel name\t: AMD EPYC 7B13 64-Core Processor\nflags\t: fpu\n"
    assert parse_cpu_model(cpuinfo) == "AMD EPYC 7B13 64-Core Processor"


def test_parses_total_memory() -> None:
    assert parse_mem_total_gib("MemTotal:       65536000 kB\nMemFree: 1 kB\n") == pytest.approx(
        62.5
    )


def test_missing_fields_give_none() -> None:
    assert parse_cpu_model("") is None
    assert parse_mem_total_gib("") is None


def test_collect_returns_a_snapshot_without_a_gpu_stack() -> None:
    info = collect_env_info(["numpy"])
    assert isinstance(info, EnvInfo)
    assert info.packages["numpy"]
