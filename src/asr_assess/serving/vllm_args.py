"""`vllm serve` command-line arguments from a VLLMServeConfig (one YAML per journey step)."""

from asr_assess.core.config import VLLMServeConfig

_POSITIONAL = {"model", "port", "extra_args"}


def vllm_args(cfg: VLLMServeConfig) -> list[str]:
    """``serve <model> --port N`` plus one flag per set option, then ``extra_args``."""
    args = ["serve", cfg.model, "--port", str(cfg.port)]
    for name, value in cfg.model_dump(exclude=_POSITIONAL).items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is not None and value is not False:
            args += [flag, str(value)]
    return args + list(cfg.extra_args)
