"""Build the engine an engine config describes; the only place that knows the backends."""

from asr_assess.core.config import HFEngineConfig, VLLMHTTPEngineConfig
from asr_assess.inference.engine import ASREngine


def make_engine(cfg: HFEngineConfig | VLLMHTTPEngineConfig) -> ASREngine:
    """Load the configured backend (heavy imports happen here, not at module import)."""
    match cfg:
        case HFEngineConfig():
            from asr_assess.inference.hf_engine import HFEngine

            return HFEngine(cfg)
        case VLLMHTTPEngineConfig():
            from asr_assess.inference.vllm_engine import VLLMHTTPEngine

            return VLLMHTTPEngine(cfg)
        case _:
            raise ValueError(f"Unknown engine kind: {cfg.kind!r}")
