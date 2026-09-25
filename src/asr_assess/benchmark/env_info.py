"""Hardware and software snapshot stored next to every benchmark result."""

import os
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from asr_assess.core.run_record import package_versions, run_command

_NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader,nounits"]  # fmt: skip
_KIB_PER_GIB = 1024 * 1024


@dataclass(frozen=True)
class GpuInfo:
    """One GPU as reported by nvidia-smi."""

    name: str
    driver: str
    memory_total_mib: int


@dataclass(frozen=True)
class EnvInfo:
    """Where a benchmark ran."""

    gpus: list[GpuInfo]
    cuda_runtime: str | None
    cpu_model: str | None
    cpu_count: int | None
    ram_gib: float | None
    platform: str
    python: str
    packages: dict[str, str | None]


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    """GPUs from ``nvidia-smi --query-gpu=name,driver_version,memory.total`` CSV output."""
    gpus = []
    for line in text.strip().splitlines():
        name, driver, memory = (field.strip() for field in line.split(","))
        gpus.append(GpuInfo(name, driver, int(float(memory))))
    return gpus


def parse_cpu_model(cpuinfo: str) -> str | None:
    """The first ``model name`` in /proc/cpuinfo."""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def parse_mem_total_gib(meminfo: str) -> float | None:
    """Total RAM in GiB from /proc/meminfo."""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / _KIB_PER_GIB
    return None


def collect_env_info(packages: Iterable[str]) -> EnvInfo:
    """Snapshot GPUs, CUDA runtime, CPU, RAM, OS, Python and package versions."""
    return EnvInfo(
        gpus=parse_nvidia_smi(run_command(_NVIDIA_SMI) or ""),
        cuda_runtime=_cuda_runtime(),
        cpu_model=parse_cpu_model(_read(Path("/proc/cpuinfo"))),
        cpu_count=os.cpu_count(),
        ram_gib=parse_mem_total_gib(_read(Path("/proc/meminfo"))),
        platform=platform.platform(),
        python=platform.python_version(),
        packages=package_versions(packages),
    )


def _cuda_runtime() -> str | None:
    try:
        import torch  # only present with the train/serve extras
    except ImportError:
        return None
    version: str | None = torch.version.cuda
    return version


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
