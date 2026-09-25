"""Turn manifest entries into engine requests; shared by evaluation, benchmarking and load tests."""

from collections.abc import Sequence
from pathlib import Path

from asr_assess.core.audio import load_audio
from asr_assess.core.config import LanguageHint
from asr_assess.core.manifest import ManifestEntry
from asr_assess.inference.engine import AudioRequest


def to_requests(
    entries: Sequence[ManifestEntry], sample_rate: int, language_hint: LanguageHint
) -> list[AudioRequest]:
    """Load each clip; force the manifest language only when ``language_hint`` is "manifest"."""
    requests = []
    for entry in entries:
        path = Path(entry.audio)
        if not path.is_file():
            raise FileNotFoundError(
                f"Audio not found: {path} (paths are relative to the repo root)"
            )
        language = entry.language if language_hint == "manifest" else None
        requests.append(AudioRequest(entry.audio, load_audio(path, sample_rate), language))
    return requests
