"""HF Trainer wiring for decoder-only LoRA, plus the training summary written for the report."""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from asr_assess.core.config import LoraTrainConfig
from asr_assess.core.manifest import ManifestEntry, read_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.training.collator import ASRCollator, Example, load_examples
from asr_assess.training.diagnostics import ParamCount, count_parameters
from asr_assess.training.lora import apply_lora

log = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class TrainOutcome:
    """What a finished run produced; everything the fine-tuning report quotes."""

    best_checkpoint: str | None
    best_metric: float | None
    global_step: int
    train_runtime_s: float
    samples_per_second: float
    peak_vram_gib: float
    lora_modules: int
    params: ParamCount
    log_history: list[dict[str, Any]]
    adapter_dir: str


def training_kwargs(cfg: LoraTrainConfig, out_dir: Path, smoke: bool) -> dict[str, Any]:
    """``TrainingArguments`` keywords for ``cfg``; smoke runs a few steps and evaluates once."""
    o = cfg.optim
    kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "learning_rate": o.learning_rate,
        "lr_scheduler_type": o.lr_scheduler_type,
        "warmup_steps": o.warmup_ratio,  # Transformers 5: a float in [0, 1) is a ratio
        "per_device_train_batch_size": o.per_device_batch_size,
        "per_device_eval_batch_size": o.per_device_batch_size,
        "gradient_accumulation_steps": o.gradient_accumulation_steps,
        "num_train_epochs": o.num_epochs,
        "bf16": o.bf16,
        "gradient_checkpointing": o.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": o.eval_strategy,
        "save_strategy": o.eval_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": o.metric_for_best_model,
        "greater_is_better": not o.metric_for_best_model.endswith("loss"),
        "save_total_limit": o.save_total_limit,
        "logging_steps": o.logging_steps,
        "seed": cfg.seed,
        "dataloader_num_workers": o.dataloader_num_workers,
        "remove_unused_columns": False,  # the collator needs the raw Example objects
        "report_to": "none",
    }
    if smoke:
        steps = cfg.smoke.max_steps
        kwargs |= {"max_steps": steps, "eval_strategy": "steps", "save_strategy": "steps",
                   "eval_steps": steps, "save_steps": steps, "logging_steps": 1}  # fmt: skip
    return kwargs


def train(
    cfg: LoraTrainConfig,
    train_set: list[Example],
    dev_set: list[Example],
    out_dir: Path,
    smoke: bool,
) -> TrainOutcome:
    """Fine-tune, keep the best epoch by dev loss, and save its adapter to ``out_dir / "best"``."""
    import torch
    import transformers

    hf: Any = transformers
    dtype = torch.bfloat16 if cfg.optim.bf16 else torch.float32
    processor = hf.AutoProcessor.from_pretrained(cfg.model_id)
    base = hf.Qwen3ASRForConditionalGeneration.from_pretrained(
        cfg.model_id, dtype=dtype, attn_implementation=cfg.attn_implementation
    )
    model, lora_modules = apply_lora(base, cfg.lora)
    params = count_parameters((n, p.numel(), p.requires_grad) for n, p in model.named_parameters())
    log.info("Trainable %d / %d parameters (%.2f%%)", params.trainable, params.total,
             params.trainable_pct)  # fmt: skip
    trainer = hf.Trainer(
        model=model,
        args=hf.TrainingArguments(**training_kwargs(cfg, out_dir, smoke)),
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=ASRCollator(processor, dtype),
        processing_class=processor,
    )
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak = torch.cuda.max_memory_allocated() / _BYTES_PER_GIB if cuda else 0.0
    adapter_dir = out_dir / "best"
    trainer.save_model(str(adapter_dir))  # the best epoch: load_best_model_at_end restored it
    return TrainOutcome(
        best_checkpoint=trainer.state.best_model_checkpoint,
        best_metric=trainer.state.best_metric,
        global_step=trainer.state.global_step,
        train_runtime_s=float(result.metrics["train_runtime"]),
        samples_per_second=float(result.metrics["train_samples_per_second"]),
        peak_vram_gib=peak,
        lora_modules=lora_modules,
        params=params,
        log_history=list(trainer.state.log_history),
        adapter_dir=str(adapter_dir),
    )


def write_train_summary(
    results_dir: Path, record: RunRecord, outcome: TrainOutcome, data: dict[str, Any]
) -> None:
    """Write train_summary.json (provenance + outcome) and log_history.jsonl (loss curve)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    body = asdict(outcome)
    history = body.pop("log_history")
    body["params"]["trainable_pct"] = outcome.params.trainable_pct
    summary = {"record": record.model_dump(mode="json"), "data": data, **body}
    (results_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (results_dir / "log_history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")


def run_training(
    cfg: LoraTrainConfig, run_name: str, smoke: bool, record: RunRecord
) -> TrainOutcome:
    """Check the run is new, load train/dev, train, and write the summary."""
    ckpt_dir, results_dir = cfg.output_dir / run_name, cfg.results_dir / run_name
    for directory in (ckpt_dir, results_dir):
        if directory.exists():
            raise FileExistsError(f"{directory} exists; runs are never overwritten")
    train_entries, dev_entries = read_manifest(cfg.train_manifest), read_manifest(cfg.dev_manifest)
    if smoke:
        train_entries = train_entries[: cfg.smoke.train_clips]
        dev_entries = dev_entries[: cfg.smoke.dev_clips]
    if not train_entries or not dev_entries:
        raise ValueError(f"Empty train or dev manifest ({cfg.train_manifest}, {cfg.dev_manifest})")
    outcome = train(cfg, load_examples(train_entries, cfg.sample_rate),
                    load_examples(dev_entries, cfg.sample_rate), ckpt_dir, smoke)  # fmt: skip
    data = {**_describe("train", train_entries), **_describe("dev", dev_entries)}
    write_train_summary(results_dir, record, outcome, data)
    log.info("Best dev loss %s at %s -> %s", outcome.best_metric, outcome.best_checkpoint,
             outcome.adapter_dir)  # fmt: skip
    return outcome


def _describe(split: str, entries: Sequence[ManifestEntry]) -> dict[str, Any]:
    minutes = round(sum(e.duration for e in entries) / 60, 2)
    return {f"{split}_clips": len(entries), f"{split}_minutes": minutes}
