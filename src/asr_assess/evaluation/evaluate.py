"""Run an ASR engine over a manifest, save every hypothesis and the WER/CER report."""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from asr_assess.core.config import EvalConfig, LanguageHint
from asr_assess.core.manifest import ManifestEntry
from asr_assess.core.run_record import RunRecord
from asr_assess.evaluation.scoring import EvalReport, Scored, build_report, score_utterance
from asr_assess.inference.engine import ASREngine, AudioRequest, Transcript, check_alignment
from asr_assess.inference.requests import to_requests

log = logging.getLogger(__name__)


def run_batches(
    engine: ASREngine, requests: Sequence[AudioRequest], batch_size: int
) -> list[Transcript]:
    """Transcribe ``requests`` in order, ``batch_size`` at a time."""
    transcripts: list[Transcript] = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        out = engine.transcribe(batch)
        check_alignment(batch, out)
        transcripts.extend(out)
        log.debug("Transcribed %d / %d", len(transcripts), len(requests))
    return transcripts


def evaluate(
    engine: ASREngine,
    entries: Sequence[ManifestEntry],
    cfg: EvalConfig,
    language_hint: LanguageHint,
    out_dir: Path,
    record: RunRecord,
) -> EvalReport:
    """Transcribe ``entries``, score them, and write hypotheses.jsonl and metrics.json."""
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    requests = to_requests(entries, cfg.sample_rate, language_hint)
    transcripts = run_batches(engine, requests, cfg.batch_size)
    items = [score_utterance(e, t) for e, t in zip(entries, transcripts, strict=True)]
    report = build_report(items, cfg.bootstrap)
    out_dir.mkdir(parents=True)  # only once there is something to write
    _write_hypotheses(out_dir / "hypotheses.jsonl", items)
    truncated = sum(i.hit_token_limit for i in items)
    if truncated:
        log.warning("%d utterances hit max_new_tokens and may be truncated", truncated)
    metrics = {
        "record": record.model_dump(mode="json"),
        "engine": engine.name,
        "n": len(items),
        "truncated": truncated,
        "report": asdict(report),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    log.info("WER %s over %d utterances -> %s", report.overall.wer.value, len(items), out_dir)
    return report


def _write_hypotheses(path: Path, items: Sequence[Scored]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = asdict(item)
            words, chars = row.pop("words"), row.pop("chars")
            row |= {"word_errors": words["errors"], "ref_words": words["ref_len"],
                    "char_errors": chars["errors"], "ref_chars": chars["ref_len"]}  # fmt: skip
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
