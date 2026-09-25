"""Tests for dataset sampling, group-held-out splitting and manifest writing."""

import io
import json
import random
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_assess.core.audio import Bucket
from asr_assess.core.config import DataConfig
from asr_assess.core.manifest import read_manifest
from asr_assess.core.run_record import RunRecord
from asr_assess.data.build import (
    build_dataset,
    clip_filename,
    is_usable,
    select_by_minutes,
    split_groups,
)
from asr_assess.data.sources import Candidate

BUCKETS = (Bucket("2-5", 2.0, 5.0), Bucket("5-15", 5.0, 15.0), Bucket("15-30", 15.0, 30.0))
SHARES = {"2-5": 0.3, "5-15": 0.45, "15-30": 0.25}


def cand(key: str, duration: float, group: str = "g", text: str = "satu dua tiga") -> Candidate:
    return Candidate(key=key, text=text, group=group, duration_hint=duration)


def pool(n_groups: int, per_group: int, durations: Sequence[float]) -> list[Candidate]:
    return [
        cand(f"g{g}-{i}", durations[(g * per_group + i) % len(durations)], group=f"g{g}")
        for g in range(n_groups)
        for i in range(per_group)
    ]


class TestIsUsable:
    @pytest.mark.parametrize(
        ("duration", "ok"), [(1.9, False), (2.0, True), (30.0, True), (31, False)]
    )
    def test_duration_bounds(self, duration: float, ok: bool) -> None:
        assert is_usable(cand("a", duration), 2.0, 30.0, min_words=2) is ok

    def test_min_words(self) -> None:
        assert not is_usable(cand("a", 3.0, text="so"), 2.0, 30.0, min_words=2)


class TestSplitGroups:
    def test_no_group_on_both_sides(self) -> None:
        train, evaluation = split_groups(pool(20, 5, [4.0]), 60.0, random.Random(0), min_groups=1)
        assert {c.group for c in train}.isdisjoint({c.group for c in evaluation})
        assert len(train) + len(evaluation) == 100

    def test_eval_side_reaches_target(self) -> None:
        _, evaluation = split_groups(pool(20, 5, [4.0]), 60.0, random.Random(0), min_groups=1)
        assert sum(c.duration_hint for c in evaluation) >= 60.0

    def test_is_seeded(self) -> None:
        a = split_groups(pool(20, 5, [4.0]), 60.0, random.Random(3), min_groups=1)
        assert split_groups(pool(20, 5, [4.0]), 60.0, random.Random(3), min_groups=1) == a

    def test_fails_when_eval_would_take_every_group(self) -> None:
        with pytest.raises(ValueError, match="enough"):
            split_groups(pool(2, 1, [4.0]), 60.0, random.Random(0), min_groups=1)

    def test_takes_at_least_min_groups_even_if_one_group_fills_the_target(self) -> None:
        big = [cand(f"big-{i}", 20.0, group="big") for i in range(10)]
        small = [cand(f"s{g}-{i}", 4.0, group=f"s{g}") for g in range(6) for i in range(2)]
        for seed in range(10):
            _, evaluation = split_groups(big + small, 60.0, random.Random(seed), min_groups=3)
            assert len({c.group for c in evaluation}) >= 3

    def test_counts_each_group_only_up_to_the_cap(self) -> None:
        big = [cand(f"big-{i}", 20.0, group="big") for i in range(10)]
        small = [cand(f"s{g}-{i}", 4.0, group=f"s{g}") for g in range(20) for i in range(2)]
        for seed in range(10):
            _, evaluation = split_groups(
                big + small, 60.0, random.Random(seed), min_groups=1, max_group_s=20.0
            )
            per_group: dict[str, float] = {}
            for c in evaluation:
                per_group[c.group] = per_group.get(c.group, 0.0) + c.duration_hint
            assert sum(min(v, 20.0) for v in per_group.values()) >= 60.0

    def test_min_groups_must_leave_a_train_side(self) -> None:
        with pytest.raises(ValueError, match="enough"):
            split_groups(pool(3, 5, [4.0]), 10.0, random.Random(0), min_groups=3)


class TestSelectByMinutes:
    def test_reaches_target_without_overshooting_by_more_than_one_clip(self) -> None:
        chosen = select_by_minutes(
            pool(10, 30, [3.0, 8.0, 20.0]),
            300.0,
            BUCKETS,
            SHARES,
            random.Random(0),
            max_group_s=None,
        )
        total = sum(c.duration_hint for c in chosen)
        assert 300.0 <= total < 300.0 + 20.0

    def test_follows_bucket_shares_when_available(self) -> None:
        chosen = select_by_minutes(
            pool(10, 30, [3.0, 8.0, 20.0]),
            600.0,
            BUCKETS,
            SHARES,
            random.Random(0),
            max_group_s=None,
        )
        long_s = sum(c.duration_hint for c in chosen if c.duration_hint >= 15)
        assert long_s == pytest.approx(0.25 * 600.0, abs=20.0)

    def test_tops_up_from_other_buckets_when_one_is_empty(self) -> None:
        chosen = select_by_minutes(
            pool(10, 30, [3.0, 8.0]), 300.0, BUCKETS, SHARES, random.Random(0), max_group_s=None
        )
        assert sum(c.duration_hint for c in chosen) >= 300.0

    def test_returns_everything_when_pool_is_short(self) -> None:
        small = pool(1, 3, [3.0])
        assert sorted(
            select_by_minutes(small, 300.0, BUCKETS, SHARES, random.Random(0), max_group_s=None),
            key=str,
        ) == sorted(small, key=str)

    def test_no_duplicates(self) -> None:
        chosen = select_by_minutes(
            pool(10, 30, [3.0, 8.0, 20.0]),
            600.0,
            BUCKETS,
            SHARES,
            random.Random(0),
            max_group_s=None,
        )
        assert len({c.key for c in chosen}) == len(chosen)


class TestGroupCap:
    def test_no_group_exceeds_the_cap(self) -> None:
        big = [cand(f"big-{i}", 3.0, group="big") for i in range(40)]
        others = [cand(f"o{g}-{i}", 3.0, group=f"o{g}") for g in range(4) for i in range(10)]
        chosen = select_by_minutes(
            big + others, 120.0, BUCKETS, SHARES, random.Random(0), max_group_s=40.0
        )
        per_group: dict[str, float] = {}
        for c in chosen:
            per_group[c.group] = per_group.get(c.group, 0.0) + c.duration_hint
        assert max(per_group.values()) <= 40.0
        assert sum(per_group.values()) >= 120.0

    def test_cap_also_limits_the_top_up(self) -> None:
        only_big = [cand(f"big-{i}", 5.0, group="big") for i in range(20)]
        chosen = select_by_minutes(
            only_big, 60.0, BUCKETS, SHARES, random.Random(0), max_group_s=30.0
        )
        assert sum(c.duration_hint for c in chosen) <= 30.0


def test_clip_filename_is_flat_and_safe() -> None:
    assert (
        clip_filename("prepared-pseudolabel-chunks/3037-0.mp3")
        == "prepared-pseudolabel-chunks_3037-0.wav"
    )


# ---------------------------------------------------------------- end to end with fakes


class FakeSource:
    def __init__(self, candidates: list[Candidate], sample_rate: int = 16000) -> None:
        self._candidates = candidates
        self._rate = sample_rate
        self.fetched: list[str] = []

    def candidates(self) -> list[Candidate]:
        return list(self._candidates)

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        by_key = {c.key: c for c in self._candidates}
        for key in keys:
            self.fetched.append(key)
            buffer = io.BytesIO()
            samples = np.zeros(int(by_key[key].duration_hint * self._rate), dtype=np.float32)
            sf.write(buffer, samples, self._rate, format="WAV")
            yield key, buffer.getvalue()


def small_config(output_dir: Path) -> DataConfig:
    fleurs = {"kind": "fleurs", "repo": "r", "revision": "v", "config": "c"}
    return DataConfig.model_validate(
        {
            "seed": 7,
            "sample_rate": 16000,
            "min_duration_s": 2.0,
            "max_duration_s": 30.0,
            "min_words": 2,
            "buckets": [{"name": b.name, "min_s": b.min_s, "max_s": b.max_s} for b in BUCKETS],
            "bucket_shares": SHARES,
            "eval_pool_factor": 2.0,
            "eval_min_groups": 1,
            "eval_max_group_share": 1.0,
            "output_dir": str(output_dir),
            "sources": {
                "shared": {**fleurs, "split": "a"},
                "read_train": {**fleurs, "split": "b"},
                "read_eval": {**fleurs, "split": "c"},
                "ctrl": {**fleurs, "split": "d"},
            },
            "categories": [
                {
                    "name": "mixed",
                    "language": None,
                    "train": {"source": "shared", "minutes": 1.0},
                    "eval": {"source": "shared", "minutes": 0.5},
                },
                {
                    "name": "read",
                    "language": "Malay",
                    "train": {"source": "read_train", "minutes": 0.5},
                    "eval": {"source": "read_eval", "minutes": 0.5},
                },
            ],
            "control": {"source": "ctrl", "minutes": 0.2, "language": "English"},
        }
    )


def record() -> RunRecord:
    return RunRecord(
        git_commit="abc",
        git_dirty=False,
        config={},
        config_hash="h",
        gpu_name=None,
        python_version="3.12",
        package_versions={},
        timestamp=datetime(2026, 9, 25, tzinfo=UTC),
    )


def fake_sources() -> dict[str, FakeSource]:
    return {
        "shared": FakeSource(pool(30, 4, [3.0, 7.0, 16.0, 1.0])),
        "read_train": FakeSource(pool(10, 3, [6.0, 9.0]), sample_rate=8000),
        "read_eval": FakeSource(pool(10, 3, [4.0, 12.0])),
        "ctrl": FakeSource(pool(5, 3, [5.0])),
    }


class TestBuildDataset:
    def test_writes_manifests_audio_and_stats(self, tmp_path: Path) -> None:
        cfg = small_config(tmp_path)
        build_dataset(cfg, fake_sources(), record(), dry_run=False)

        train = read_manifest(tmp_path / "manifests" / "train.jsonl")
        evaluation = read_manifest(tmp_path / "manifests" / "eval.jsonl")
        control = read_manifest(tmp_path / "manifests" / "control.jsonl")

        assert sum(e.duration for e in train) >= 90.0
        assert all(Path(e.audio).exists() for e in [*train, *evaluation, *control])
        assert all(2.0 <= e.duration <= 30.0 and e.bucket in SHARES for e in train)
        assert {e.language for e in train} == {None, "Malay"}
        assert {e.language for e in control} == {"English"}
        stats = json.loads((tmp_path / "manifests" / "stats.json").read_text(encoding="utf-8"))
        assert stats["seed"] == 7
        assert stats["record"]["git_commit"] == "abc"
        assert set(stats["splits"]) == {"train", "eval", "control"}

    def test_audio_is_resampled_to_target_rate(self, tmp_path: Path) -> None:
        build_dataset(small_config(tmp_path), fake_sources(), record(), dry_run=False)
        entry = next(
            e for e in read_manifest(tmp_path / "manifests" / "train.jsonl") if "read/" in e.source
        )
        assert sf.info(entry.audio).samplerate == 16000

    def test_shared_source_groups_never_leak_into_eval(self, tmp_path: Path) -> None:
        build_dataset(small_config(tmp_path), fake_sources(), record(), dry_run=False)

        def groups(split: str) -> set[str]:
            entries = read_manifest(tmp_path / "manifests" / f"{split}.jsonl")
            return {
                Path(e.audio).stem.split("-")[0] for e in entries if e.source.startswith("mixed/")
            }

        assert groups("train").isdisjoint(groups("eval"))

    def test_is_reproducible(self, tmp_path: Path) -> None:
        build_dataset(small_config(tmp_path / "a"), fake_sources(), record(), dry_run=False)
        build_dataset(small_config(tmp_path / "b"), fake_sources(), record(), dry_run=False)
        a = [e.text for e in read_manifest(tmp_path / "a" / "manifests" / "train.jsonl")]
        b = [e.text for e in read_manifest(tmp_path / "b" / "manifests" / "train.jsonl")]
        keys_a = [
            Path(e.audio).name for e in read_manifest(tmp_path / "a" / "manifests" / "train.jsonl")
        ]
        keys_b = [
            Path(e.audio).name for e in read_manifest(tmp_path / "b" / "manifests" / "train.jsonl")
        ]
        assert (a, keys_a) == (b, keys_b)

    def test_dry_run_fetches_and_writes_nothing(self, tmp_path: Path) -> None:
        sources = fake_sources()
        build_dataset(small_config(tmp_path), sources, record(), dry_run=True)
        assert not (tmp_path / "manifests").exists()
        assert all(not s.fetched for s in sources.values())


class TestEvalPoolFactorOverride:
    def config(self, tmp_path: Path, override: float | None) -> DataConfig:
        raw = small_config(tmp_path).model_dump(mode="json")
        raw["eval_pool_factor"] = 3.0
        raw["categories"][0]["train"]["minutes"] = 0.5
        raw["categories"][0]["eval"]["minutes"] = 0.25
        raw["categories"][0]["eval_pool_factor"] = override
        return DataConfig.model_validate(raw)

    def scarce_sources(self) -> dict[str, FakeSource]:
        # 6 speakers x 10 s: a 3x eval pool (45 s) leaves train only 15 s of its 30 s quota.
        return {**fake_sources(), "shared": FakeSource(pool(6, 2, [4.0, 6.0]))}

    def train_seconds(self, cfg: DataConfig) -> float:
        plans = build_dataset(cfg, self.scarce_sources(), record(), dry_run=True)
        return next(p.planned_s for p in plans if (p.split, p.category) == ("train", "mixed"))

    def test_default_factor_starves_a_small_source(self, tmp_path: Path) -> None:
        assert self.train_seconds(self.config(tmp_path, None)) < 30.0

    def test_category_override_leaves_train_its_quota(self, tmp_path: Path) -> None:
        assert self.train_seconds(self.config(tmp_path, 1.0)) >= 30.0


class TestEvalSpread:
    def test_eval_plan_spreads_across_speakers(self, tmp_path: Path) -> None:
        raw = small_config(tmp_path).model_dump(mode="json")
        raw["eval_min_groups"], raw["eval_max_group_share"] = 3, 0.4
        cfg = DataConfig.model_validate(raw)
        # One speaker could fill the whole eval quota alone.
        big = [cand(f"big-{i}", 6.0, group="big") for i in range(30)]
        small = [cand(f"s{g}-{i}", 6.0, group=f"s{g}") for g in range(8) for i in range(4)]
        sources = {**fake_sources(), "shared": FakeSource(big + small)}
        plans = build_dataset(cfg, sources, record(), dry_run=True)
        evaluation = next(p for p in plans if (p.split, p.category) == ("eval", "mixed"))
        per_group: dict[str, float] = {}
        for c in evaluation.clips:
            per_group[c.group] = per_group.get(c.group, 0.0) + c.duration_hint
        assert len(per_group) >= 3
        assert max(per_group.values()) <= 0.4 * evaluation.target_s
