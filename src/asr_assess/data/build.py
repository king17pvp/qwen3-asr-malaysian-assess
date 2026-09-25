"""Sample, clean, split (held out by speaker/video) and write the train/eval/control manifests."""

import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from asr_assess.core.audio import Bucket, assign_bucket, decode_audio, duration_s, write_wav
from asr_assess.core.config import CategorySpec, DataConfig, Quota
from asr_assess.core.manifest import ManifestEntry, write_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.core.text import build_qwen_target
from asr_assess.data.sources import Candidate, ClipSource

log = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# A quota met to within this fraction is not reported as a shortfall.
_SHORTFALL_TOLERANCE = 0.98


@dataclass(frozen=True)
class SplitPlan:
    """Clips chosen for one (split, category) before any audio is fetched."""

    split: str
    category: str
    language: str | None
    source: str
    target_s: float
    clips: list[Candidate]

    @property
    def planned_s(self) -> float:
        """Total duration of the chosen clips, from metadata hints."""
        return sum(c.duration_hint for c in self.clips)


@dataclass(frozen=True)
class SplitStats:
    """What was written for one split."""

    clips: int
    minutes: float
    minutes_by_bucket: dict[str, float]
    minutes_by_source: dict[str, float]


def is_usable(candidate: Candidate, min_s: float, max_s: float, min_words: int) -> bool:
    """Whether a candidate's duration hint and transcript length pass the filters."""
    in_range = min_s <= candidate.duration_hint <= max_s
    return in_range and len(candidate.text.split()) >= min_words


def split_groups(
    candidates: Sequence[Candidate],
    eval_target_s: float,
    rng: random.Random,
    min_groups: int,
    max_group_s: float | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Move whole random groups to eval until it holds ``eval_target_s`` and at least
    ``min_groups`` groups; the rest is train.

    With ``max_group_s``, a group counts only up to that many seconds, matching the cap that
    ``select_by_minutes`` later applies, so the eval side can actually fill its quota.
    """
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    groups = sorted(by_group)
    rng.shuffle(groups)
    eval_groups: set[str] = set()
    held_s = 0.0
    for group in groups:
        if held_s >= eval_target_s and len(eval_groups) >= min_groups:
            break
        eval_groups.add(group)
        group_s = sum(c.duration_hint for c in by_group[group])
        held_s += group_s if max_group_s is None else min(group_s, max_group_s)
    enough = held_s >= eval_target_s and len(eval_groups) >= min_groups
    if not enough or len(eval_groups) == len(groups):
        raise ValueError(
            f"Not enough groups to hold out {eval_target_s:.0f} s over {min_groups}+ groups "
            "and keep a train side"
        )
    train = [c for c in candidates if c.group not in eval_groups]
    return train, [c for c in candidates if c.group in eval_groups]


def select_by_minutes(
    pool: Sequence[Candidate],
    target_s: float,
    buckets: Sequence[Bucket],
    shares: Mapping[str, float],
    rng: random.Random,
    max_group_s: float | None,
) -> list[Candidate]:
    """Random clips filling each bucket's share of ``target_s``, topped up from any bucket.

    With ``max_group_s``, no group (speaker/video) contributes more than that many seconds.
    """
    order = sorted(pool, key=lambda c: c.key)
    rng.shuffle(order)
    chosen: list[Candidate] = []
    group_s: dict[str, float] = defaultdict(float)

    def take_if(candidate: Candidate, room_s: float) -> float:
        d = candidate.duration_hint
        capped = max_group_s is not None and group_s[candidate.group] + d > max_group_s
        if d > room_s or capped:
            return 0.0
        chosen.append(candidate)
        group_s[candidate.group] += d
        return d

    for bucket in buckets:
        quota, got = shares[bucket.name] * target_s, 0.0
        for candidate in order:
            if assign_bucket(candidate.duration_hint, buckets) == bucket.name:
                got += take_if(candidate, quota - got)
    taken = {c.key for c in chosen}
    total = sum(group_s.values())
    for candidate in order:
        if total >= target_s:
            break
        if candidate.key not in taken:
            total += take_if(candidate, float("inf"))
    return chosen


def clip_filename(key: str) -> str:
    """A flat, filesystem-safe WAV name derived from a source key."""
    stem = PurePosixPath(key).with_suffix("").as_posix()
    return _UNSAFE.sub("_", stem).strip("_") + ".wav"


def plan_dataset(cfg: DataConfig, sources: Mapping[str, ClipSource]) -> list[SplitPlan]:
    """Choose every split's clips from metadata alone; each category has its own seeded RNG."""
    usable = {name: _usable(cfg, source) for name, source in sources.items()}
    plans: list[SplitPlan] = []
    for cat in cfg.categories:
        rng = random.Random(f"{cfg.seed}:{cat.name}")
        train_pool, eval_pool = _category_pools(cfg, cat, usable, rng)
        plans.append(_plan(cfg, "train", cat.name, cat.language, cat.train, train_pool, rng))
        plans.append(_plan(cfg, "eval", cat.name, cat.language, cat.eval, eval_pool, rng))
    control, rng = cfg.control, random.Random(f"{cfg.seed}:control")
    pool = usable[control.source]
    plans.append(_plan(cfg, "control", "control", control.language, control, pool, rng))
    return plans


def _usable(cfg: DataConfig, source: ClipSource) -> list[Candidate]:
    return [
        c
        for c in source.candidates()
        if is_usable(c, cfg.min_duration_s, cfg.max_duration_s, cfg.min_words)
    ]


def _category_pools(
    cfg: DataConfig,
    cat: CategorySpec,
    usable: Mapping[str, list[Candidate]],
    rng: random.Random,
) -> tuple[list[Candidate], list[Candidate]]:
    if cat.train.source != cat.eval.source:
        return usable[cat.train.source], usable[cat.eval.source]
    factor = cat.eval_pool_factor or cfg.eval_pool_factor
    eval_pool_s = cat.eval.minutes * 60 * factor
    max_group_s = cat.eval.minutes * 60 * cfg.eval_max_group_share
    pool = usable[cat.train.source]
    return split_groups(pool, eval_pool_s, rng, cfg.eval_min_groups, max_group_s)


def _plan(
    cfg: DataConfig,
    split: str,
    category: str,
    language: str | None,
    quota: Quota,
    pool: Sequence[Candidate],
    rng: random.Random,
) -> SplitPlan:
    buckets = [b.to_bucket() for b in cfg.buckets]
    target_s = quota.minutes * 60
    # Eval is capped per group so one talkative speaker/video cannot dominate the WER.
    max_group_s = target_s * cfg.eval_max_group_share if split == "eval" else None
    clips = select_by_minutes(pool, target_s, buckets, cfg.bucket_shares, rng, max_group_s)
    return SplitPlan(split, category, language, quota.source, target_s, clips)


def materialize(plan: SplitPlan, source: ClipSource, cfg: DataConfig) -> list[ManifestEntry]:
    """Fetch, resample, re-check and write one plan's clips as 16-bit mono WAVs."""
    buckets = [b.to_bucket() for b in cfg.buckets]
    text_of = {c.key: c.text for c in plan.clips}
    out_dir = cfg.output_dir / "audio" / plan.split / plan.category
    entries = []
    for key, blob in source.fetch(sorted(text_of)):
        samples = decode_audio(blob, cfg.sample_rate)
        duration = duration_s(samples, cfg.sample_rate)
        bucket = assign_bucket(duration, buckets)
        if bucket is None or not cfg.min_duration_s <= duration <= cfg.max_duration_s:
            log.warning("Dropping %s: decoded duration %.2f s is out of range", key, duration)
            continue
        path = out_dir / clip_filename(key)
        write_wav(path, samples, cfg.sample_rate)
        entries.append(
            ManifestEntry(
                audio=path.as_posix(),
                text=build_qwen_target(plan.language, text_of[key]),
                duration=round(duration, 3),
                bucket=bucket,
                source=f"{plan.category}/{plan.source}",
            )
        )
    return sorted(entries, key=lambda e: e.audio)


def summarize(entries: Sequence[ManifestEntry]) -> SplitStats:
    """Clip count and minutes, overall and per bucket and source."""
    by_bucket: dict[str, float] = defaultdict(float)
    by_source: dict[str, float] = defaultdict(float)
    for entry in entries:
        by_bucket[entry.bucket] += entry.duration / 60
        by_source[entry.source] += entry.duration / 60
    return SplitStats(
        clips=len(entries),
        minutes=round(sum(e.duration for e in entries) / 60, 2),
        minutes_by_bucket={k: round(v, 2) for k, v in sorted(by_bucket.items())},
        minutes_by_source={k: round(v, 2) for k, v in sorted(by_source.items())},
    )


def build_dataset(
    cfg: DataConfig, sources: Mapping[str, ClipSource], record: RunRecord, dry_run: bool
) -> list[SplitPlan]:
    """Plan every split, then (unless ``dry_run``) write audio, manifests and stats.json."""
    plans = plan_dataset(cfg, sources)
    _log_plans(plans)
    if dry_run:
        return plans
    by_split: dict[str, list[ManifestEntry]] = defaultdict(list)
    for plan in plans:
        by_split[plan.split].extend(materialize(plan, sources[plan.source], cfg))
    manifest_dir = cfg.output_dir / "manifests"
    for split, entries in by_split.items():
        write_manifest(manifest_dir / f"{split}.jsonl", entries)
    _write_stats(manifest_dir / "stats.json", cfg, record, plans, by_split)
    return plans


def _log_plans(plans: Sequence[SplitPlan]) -> None:
    for plan in plans:
        groups = len({c.group for c in plan.clips})
        minutes = (plan.planned_s / 60, plan.target_s / 60)
        log.info(
            "%-7s %-22s %4d clips %6.2f / %5.2f min from %d groups",
            plan.split, plan.category, len(plan.clips), *minutes, groups,
        )  # fmt: skip
        if plan.planned_s < plan.target_s * _SHORTFALL_TOLERANCE:
            log.warning("%s/%s is short of its quota", plan.split, plan.category)


def _write_stats(
    path: Path,
    cfg: DataConfig,
    record: RunRecord,
    plans: Sequence[SplitPlan],
    by_split: Mapping[str, Sequence[ManifestEntry]],
) -> None:
    stats = {
        "seed": cfg.seed,
        "record": record.model_dump(mode="json"),
        "splits": {split: asdict(summarize(entries)) for split, entries in by_split.items()},
        "groups": {f"{p.split}/{p.category}": len({c.group for c in p.clips}) for p in plans},
    }
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
