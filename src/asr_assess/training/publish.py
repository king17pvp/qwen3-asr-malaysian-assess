"""Publish a verified merged checkpoint to the Hugging Face Hub with a generated model card."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from asr_assess.core.config import LoraTrainConfig
from asr_assess.core.run_record import RunRecord

log = logging.getLogger(__name__)

LANGUAGES = ["ms", "en"]


class HubClient(Protocol):
    """The slice of ``huggingface_hub.HfApi`` a push uses."""

    def whoami(self) -> dict[str, Any]: ...
    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...
    def upload_file(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class VerifiedMerge:
    """A merged checkpoint whose weight-delta check passed, with what `merge` recorded about it."""

    checkpoint: Path
    summary: dict[str, Any]
    smoke: list[dict[str, Any]]


def hub_client() -> HubClient:
    """A Hub client using HF_TOKEN or the `hf auth login` token."""
    from huggingface_hub import HfApi

    client: HubClient = HfApi()
    return client


def verified_merge(cfg: LoraTrainConfig, run_name: str) -> VerifiedMerge:
    """The run's merged checkpoint; refuses one that `asr-assess merge` did not verify."""
    checkpoint, results = cfg.merged_dir / run_name, cfg.merge_results_dir / run_name
    summary_path = results / "merge_summary.json"
    if not checkpoint.is_dir() or not summary_path.is_file():
        raise FileNotFoundError(f"No verified merge for {run_name!r}; run `asr-assess merge` first")
    deltas = _read_json(results / "weight_deltas.json")
    if deltas.get("ok") is not True:
        raise RuntimeError(f"{results / 'weight_deltas.json'} failed its check; not publishing")
    lines = (results / "smoke.jsonl").read_text(encoding="utf-8").splitlines()
    return VerifiedMerge(checkpoint, _read_json(summary_path), [json.loads(x) for x in lines if x])


def build_model_card(
    cfg: LoraTrainConfig, merge: VerifiedMerge, repo_id: str, eval_metrics: Sequence[Path]
) -> str:
    """README.md for the Hub repo: metadata, training provenance, smoke output, optional WER."""
    meta = {
        "base_model": cfg.model_id,
        "pipeline_tag": "automatic-speech-recognition",
        "library_name": "transformers",
        "language": LANGUAGES,
        "tags": ["qwen3-asr", "lora", "malaysian"],
    }
    record = merge.summary["record"]
    lora = cfg.lora
    lines = [
        "---",
        yaml.safe_dump(meta, sort_keys=False).rstrip(),
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"[{cfg.model_id}](https://huggingface.co/{cfg.model_id}) fine-tuned on Malaysian speech "
        "with decoder-only LoRA (audio encoder frozen), merged into standalone weights: "
        "no PEFT needed to load it.",
        "",
        "## Training",
        "",
        f"- LoRA: rank {lora.rank}, alpha {lora.alpha}, dropout {lora.dropout}, "
        f"targets {', '.join(lora.target_modules)} (decoder only)",
        f"- Optimiser: lr {cfg.optim.learning_rate}, {cfg.optim.num_epochs} epochs, "
        "best epoch by dev loss",
        f"- Code: commit `{record['git_commit']}`" + (" (dirty)" if record["git_dirty"] else ""),
        f"- Merged: {record['timestamp']}; {merge.summary['changed_tensors']} tensors changed, "
        "all inside the LoRA targets",
    ]
    if eval_metrics:
        lines += ["", "## Evaluation", "", "| Run | Clips | WER [95% CI] | CER |"]
        lines += ["|---|---|---|---|", *(_metrics_row(path) for path in eval_metrics)]
    lines += ["", "## Smoke transcripts (dev)", ""]
    lines += [f"- ref: `{row['reference']}`\n  hyp: `{row['hypothesis']}`" for row in merge.smoke]
    lines += [
        "",
        "## Usage",
        "",
        f"Set `model_id: {repo_id}` in `configs/engines/hf_ft.yaml`, or serve it with vLLM:",
        "",
        "```bash",
        f"vllm serve {repo_id}",
        "```",
        "",
    ]
    return "\n".join(lines)


def run_push(
    cfg: LoraTrainConfig,
    run_name: str,
    repo_id: str,
    public: bool,
    eval_metrics: Sequence[Path],
    record: RunRecord,
    hub: HubClient,
) -> dict[str, Any]:
    """Upload the verified merge, then its model card; write push_summary.json."""
    results = cfg.push_results_dir / run_name
    if results.exists():
        raise FileExistsError(f"{results} exists; results are never overwritten")
    merge = verified_merge(cfg, run_name)
    card = build_model_card(cfg, merge, repo_id, eval_metrics)
    try:
        user = hub.whoami()["name"]
    except Exception as exc:
        raise PermissionError("No Hub login: set HF_TOKEN or run `hf auth login`") from exc
    trained = merge.summary["record"]["git_commit"]
    url = hub.create_repo(repo_id=repo_id, private=not public, exist_ok=True)
    weights = hub.upload_folder(
        repo_id=repo_id,
        folder_path=str(merge.checkpoint),
        commit_message=f"Merged LoRA run {run_name} (trained at {trained})",
    )
    commit = hub.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=f"Model card (pushed at {record.git_commit}, config {record.config_hash})",
    )
    summary = {
        "record": record.model_dump(mode="json"),
        "user": user,
        "repo_id": repo_id,
        "url": str(url),
        "private": not public,
        "checkpoint": str(merge.checkpoint),
        "weights_commit": weights.oid,
        "commit": commit.oid,
        "commit_url": commit.commit_url,
    }
    results.mkdir(parents=True)
    (results / "push_summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    log.info("Pushed %s to %s at %s", merge.checkpoint, url, commit.oid)
    return summary


def _metrics_row(path: Path) -> str:
    overall = _read_json(path)["report"]["overall"]
    wer, cer = overall["wer"], overall["cer"]
    wer_ci = f"{_pct(wer['value'])}% [{_pct(wer['low'])}, {_pct(wer['high'])}]"
    return f"| {path.parent.name} | {overall['n']} | {wer_ci} | {_pct(cer['value'])}% |"


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.2f}"


def _read_json(path: Path) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return body
