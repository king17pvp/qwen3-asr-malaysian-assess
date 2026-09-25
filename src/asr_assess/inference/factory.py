"""Build the engine an ``EngineConfig`` describes; the only place that knows the backends."""

from asr_assess.core.config import EngineConfig
from asr_assess.inference.engine import ASREngine


def make_engine(cfg: EngineConfig) -> ASREngine:
    """Load the configured backend (heavy imports happen here, not at module import)."""
    match cfg.kind:
        case "hf":
            from asr_assess.inference.hf_engine import HFEngine

            return HFEngine(cfg)
        case _:
            raise ValueError(f"Unknown engine kind: {cfg.kind!r}")
