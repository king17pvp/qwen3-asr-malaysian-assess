"""Merge the best adapter into a standalone checkpoint, prove what changed, and reload it."""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from asr_assess.core.config import EngineConfig, LoraTrainConfig
from asr_assess.core.manifest import ManifestEntry, read_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.inference.engine import ASREngine, check_alignment
from asr_assess.inference.requests import to_requests
from asr_assess.training.diagnostics import DeltaReport, summarize_deltas, weight_deltas
from asr_assess.training.lora import decoder_target_pattern

log = logging.getLogger(__name__)


def smoke_transcribe(
    engine: ASREngine, entries: Sequence[ManifestEntry], sample_rate: int
) -> list[dict[str, Any]]:
    """Transcribe ``entries`` one at a time; fail if any transcript is empty."""
    rows: list[dict[str, Any]] = []
    for entry, request in zip(entries, to_requests(entries, sample_rate, "auto"), strict=True):
        [transcript] = engine.transcribe([request])
        check_alignment([request], [transcript])
        rows.append(
            {
                "id": entry.audio,
                "reference": entry.transcript,
                "hypothesis": transcript.text,
                "language": transcript.language,
            }
        )
    empty = [row["id"] for row in rows if not row["hypothesis"].strip()]
    if empty:
        raise RuntimeError(f"Merged model returned empty transcripts for {empty}")
    return rows


def merge_adapter(model_id: str, adapter_dir: Path, out_dir: Path) -> None:
    """Fold the LoRA adapter into the base weights and save a checkpoint with no PEFT dependency."""
    import transformers
    from peft import PeftModel

    hf: Any = transformers
    # "auto" keeps the checkpoint's stored dtype, so frozen tensors are saved bit-identical.
    base = hf.Qwen3ASRForConditionalGeneration.from_pretrained(model_id, dtype="auto")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir))
    hf.AutoProcessor.from_pretrained(model_id).save_pretrained(str(out_dir))
    log.info("Merged %s into %s", adapter_dir, out_dir)


def base_snapshot(model_id: str) -> Path:
    """Local directory of the base checkpoint's weights (a local path is returned as is)."""
    if Path(model_id).is_dir():
        return Path(model_id)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))


def run_merge(
    cfg: LoraTrainConfig, engine_cfg: EngineConfig, run_name: str, record: RunRecord
) -> DeltaReport:
    """Merge, write the weight-delta proof, then reload the result and transcribe dev clips."""
    adapter = cfg.output_dir / run_name / "best"
    merged, results = cfg.merged_dir / run_name, cfg.merge_results_dir / run_name
    if not adapter.is_dir():
        raise FileNotFoundError(f"No adapter at {adapter}; run `asr-assess train` first")
    for directory in (merged, results):
        if directory.exists():
            raise FileExistsError(f"{directory} exists; results are never overwritten")
    merge_adapter(cfg.model_id, adapter, merged)
    pattern = decoder_target_pattern(cfg.lora.target_modules)
    report = summarize_deltas(weight_deltas(base_snapshot(cfg.model_id), merged), pattern)
    results.mkdir(parents=True)
    _write_json(results / "weight_deltas.json", asdict(report) | {"ok": report.ok})
    if not report.ok:
        unexpected = report.unexpected_changes[:3]
        raise RuntimeError(f"Weights changed outside the LoRA targets, or none did: {unexpected}")
    from asr_assess.inference import factory

    engine = factory.make_engine(engine_cfg.model_copy(update={"model_id": str(merged)}))
    entries = read_manifest(cfg.dev_manifest)[: cfg.smoke_clips]
    rows = smoke_transcribe(engine, entries, cfg.sample_rate)
    with (results / "smoke.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "record": record.model_dump(mode="json"),
        "adapter": str(adapter),
        "merged": str(merged),
        "changed_tensors": len(report.changed),
    }
    _write_json(results / "merge_summary.json", summary)
    return report


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
