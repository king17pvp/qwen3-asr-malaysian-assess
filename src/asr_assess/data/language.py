"""Flag Indonesian videos in Malaysian YouTube data by vocabulary.

Some upstream "Malaysian context" videos are Indonesian speech tagged as Malay. Each video is
scored on its whole transcript (~20 min), which is far more reliable than scoring single clips.
Only words that one variety uses and the other does not count as evidence; words shared with
formal Malaysian Malay (e.g. "ingin", "tinggal") were measured to cause false positives.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def _words(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# Indonesian forms, with the Malaysian equivalent where there is one.
_INDONESIAN = _words(
    "nggak", "ngga", "enggak", "gak",  # tak
    "banget", "gimana", "aja", "sih", "dong", "emang", "udah", "yaudah", "bikin", "gue", "kayak",
    "karena",  # kerana
    "uang",  # wang
    "mobil", "mobilnya",  # kereta
    "kantor",  # pejabat
    "rumah sakit",  # hospital
    "bilang",  # cakap
    "sepeda",  # basikal
    "bisa",  # boleh
)  # fmt: skip
_MALAYSIAN = _words(
    "tak", "lah", "kerana", "wang", "boleh", "macam", "cakap", "dah", "nak", "kereta",
    "pejabat", "hospital", "awak", "korang", "tu", "ni", "sebab", "mahu",
)  # fmt: skip


@dataclass(frozen=True)
class MarkerCounts:
    """Occurrences of Indonesian-only and Malaysian-only words."""

    indonesian: int
    malaysian: int


def count_markers(text: str) -> MarkerCounts:
    """Count Indonesian-only and Malaysian-only words in ``text``."""
    return MarkerCounts(len(_INDONESIAN.findall(text)), len(_MALAYSIAN.findall(text)))


def flag_indonesian(
    texts_by_video: Mapping[str, Iterable[str]], max_share: float, min_hits: int
) -> set[str]:
    """Videos whose Indonesian share of marker words exceeds ``max_share``.

    Videos with fewer than ``min_hits`` marker words are kept: too little evidence either way.
    """
    flagged = set()
    for video, texts in texts_by_video.items():
        counts = count_markers(" ".join(texts))
        hits = counts.indonesian + counts.malaysian
        if hits >= min_hits and counts.indonesian / hits > max_share:
            flagged.add(video)
    return flagged
