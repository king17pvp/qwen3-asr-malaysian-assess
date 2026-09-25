"""JSONL manifest schema shared by data building, training and evaluation."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from asr_assess.core.text import ASR_TEXT_TAG, LANGUAGE_PREFIX, parse_qwen_output


class ManifestEntry(BaseModel):
    """One utterance: a 16 kHz mono WAV and its Qwen3-ASR target text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio: str
    text: str
    duration: float = Field(gt=0)
    bucket: str
    source: str

    @field_validator("text")
    @classmethod
    def _has_qwen_prefix(cls, value: str) -> str:
        if not value.startswith(LANGUAGE_PREFIX) or ASR_TEXT_TAG not in value:
            raise ValueError(f"text must look like 'language X{ASR_TEXT_TAG}...'")
        return value

    @property
    def language(self) -> str | None:
        """Language named in the prefix, or None for ``language None``."""
        return parse_qwen_output(self.text)[0]

    @property
    def transcript(self) -> str:
        """Reference transcript without the Qwen prefix."""
        return parse_qwen_output(self.text)[1]


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read and validate a JSONL manifest, naming the offending line on error."""
    entries: list[ManifestEntry] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(ManifestEntry.model_validate_json(line))
            except ValidationError as err:
                raise ValueError(f"{path}:{line_no}: {err}") from err
    return entries


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    """Write entries as JSONL (one object per line), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")
