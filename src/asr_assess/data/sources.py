"""Dataset sources behind one interface: list candidate clips cheaply, then fetch chosen audio.

Every source reads the Hugging Face Hub selectively (parquet columns / row groups, single zip
members via HTTP range requests), so nothing close to the full datasets is downloaded.
``huggingface_hub`` and ``pyarrow`` come from the ``data`` extra and are imported lazily.
"""

import io
import json
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

import soundfile as sf

from asr_assess.core.config import (
    FleursSource,
    LibriSpeechSource,
    MesoliticaContextSource,
    MesoliticaConversationalSource,
    SourceSpec,
)
from asr_assess.core.text import strip_whisper_tokens, whisper_language_code
from asr_assess.data.language import flag_indonesian

if TYPE_CHECKING:
    from huggingface_hub import HfFileSystem

log = logging.getLogger(__name__)

_TIMESTAMP = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_VIDEO_WINDOW = re.compile(r"^(\d+-\d+)-\d+$")
# Small blocks: zip members are tens of KB, so large read-ahead would mostly be wasted.
_ZIP_BLOCK_BYTES = 256 * 1024
_WAV_HEADER_BYTES = 44


@dataclass(frozen=True)
class Candidate:
    """A clip that could be sampled, known from metadata only (no audio fetched yet)."""

    key: str
    text: str
    group: str  # speaker or video id; a group never spans train and eval
    duration_hint: float


@dataclass(frozen=True)
class IndexLine:
    """One line of prepared-pseudolabel.jsonl: a 30 s window and one Whisper decode of it."""

    audio_filename: str
    text: str


class ClipSource(Protocol):
    """Lists candidate clips, then fetches the audio bytes of the chosen ones."""

    def candidates(self) -> list[Candidate]:
        """Every usable clip with its transcript, group and approximate duration."""
        ...

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Encoded audio (any format soundfile reads) for each requested key."""
        ...


# ---------------------------------------------------------------- pure helpers


def chunk_index(audio_filename: str) -> int:
    """Index-line number N of a ``prepared-pseudolabel-chunks/N-k.mp3`` clip."""
    return int(PurePosixPath(audio_filename).stem.split("-")[0])


def video_id(index_audio_path: str) -> str:
    """YouTube video id (``worker-video``) of an ``output-audio/worker-video-window.mp3`` path."""
    match = _VIDEO_WINDOW.match(PurePosixPath(index_audio_path).stem)
    if match is None:
        raise ValueError(f"Unexpected pseudolabel audio path: {index_audio_path!r}")
    return match.group(1)


def conversational_row(audio_filename: str, prefix: str) -> int:
    """Row of the original conversational corpus that ``<prefix>N.wav`` was written from."""
    return int(PurePosixPath(audio_filename.removeprefix(prefix)).stem)


def last_timestamp(token_text: str) -> float | None:
    """The final Whisper timestamp in a token string, which approximates clip duration."""
    stamps = _TIMESTAMP.findall(token_text)
    return float(stamps[-1]) if stamps else None


def pcm_wav_duration(file_size: int, byte_rate: int) -> float:
    """Duration of a canonical PCM WAV (44-byte header) from its size, without reading it."""
    if byte_rate <= 0:
        raise ValueError("Byte rate must be positive")
    return (file_size - _WAV_HEADER_BYTES) / byte_rate


def parse_index_head(blob: bytes) -> list[IndexLine]:
    """Lines of a truncated JSONL index, dropping the partial last line.

    List position is the line number, which clip names refer to, so no line is skipped.
    """
    lines = blob.decode("utf-8", errors="ignore").split("\n")[:-1]
    return [IndexLine(r["audio_filename"], r["new_text"]) for r in map(json.loads, lines)]


# ---------------------------------------------------------------- Hub access


def _hub() -> "HfFileSystem":
    from huggingface_hub import HfFileSystem

    return HfFileSystem()  # reads HF_TOKEN from the environment when set


def _hub_path(repo: str, path: str, revision: str | None = None) -> str:
    rev = f"@{quote(revision, safe='')}" if revision else ""
    return f"datasets/{repo}{rev}/{path}"


def _read_columns(
    fs: "HfFileSystem", path: str, columns: list[str], row_group: int | None = None
) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        table = (
            parquet.read(columns=columns)
            if row_group is None
            else parquet.read_row_group(row_group, columns=columns)
        )
    columns_by_name: dict[str, list[Any]] = table.to_pydict()
    return columns_by_name


def _parquet_audio(
    fs: "HfFileSystem", path: str, key_column: str, keys: Sequence[str]
) -> Iterator[tuple[str, bytes]]:
    """Audio bytes for ``keys``, reading the audio column only of row groups that hold them."""
    import pyarrow.parquet as pq

    wanted = set(keys)
    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for group in range(parquet.num_row_groups):
            group_keys = parquet.read_row_group(group, columns=[key_column]).column(0).to_pylist()
            if wanted.isdisjoint(group_keys):
                continue
            rows = parquet.read_row_group(group, columns=[key_column, "audio"]).to_pylist()
            for row in rows:
                if row[key_column] in wanted:
                    yield row[key_column], row["audio"]["bytes"]


def _zip_names(fs: "HfFileSystem", path: str) -> list[str]:
    with fs.open(path, "rb", block_size=_ZIP_BLOCK_BYTES) as handle:
        return zipfile.ZipFile(handle).namelist()


def _zip_wav_durations(fs: "HfFileSystem", path: str, prefix: str) -> dict[str, float]:
    """Exact durations of the PCM WAVs under ``prefix``, from the zip directory's sizes.

    The byte rate is read from one member's header; every member is assumed to share it, and
    ``build.materialize`` re-checks each decoded duration anyway.
    """
    with fs.open(path, "rb", block_size=_ZIP_BLOCK_BYTES) as handle:
        archive = zipfile.ZipFile(handle)
        infos = [i for i in archive.infolist() if i.filename.startswith(prefix)]
        header = sf.info(io.BytesIO(archive.read(infos[0].filename)))
    if header.subtype != "PCM_16":
        raise ValueError(f"{path}: expected PCM_16 WAVs, found {header.subtype}")
    byte_rate = header.samplerate * header.channels * 2
    return {i.filename: pcm_wav_duration(i.file_size, byte_rate) for i in infos}


def _zip_members(
    fs: "HfFileSystem", path: str, names: Sequence[str]
) -> Iterator[tuple[str, bytes]]:
    with fs.open(path, "rb", block_size=_ZIP_BLOCK_BYTES) as handle:
        archive = zipfile.ZipFile(handle)
        for name in names:
            yield name, archive.read(name)


# ---------------------------------------------------------------- sources


class MesoliticaContext:
    """Manglish YouTube clips from mesolitica/Malaysian-STT-Whisper ``malaysian_context_v2``."""

    def __init__(self, spec: MesoliticaContextSource, fs: "HfFileSystem") -> None:
        self._spec = spec
        self._fs = fs
        self._zip_of: dict[str, str] = {}

    def candidates(self) -> list[Candidate]:
        """Clips in the index head and the configured zips, grouped by YouTube video."""
        spec = self._spec
        with self._fs.open(_hub_path(spec.index_repo, spec.index_path), "rb") as handle:
            index = parse_index_head(handle.read(spec.index_bytes))
        excluded = self._indonesian_videos(index)
        for zip_name in spec.zips:
            for member in _zip_names(self._fs, _hub_path(spec.repo, zip_name)):
                self._zip_of[member] = zip_name
        rows = _read_columns(
            self._fs,
            _hub_path(spec.repo, spec.parquet),
            ["audio_filename", "segment_timestamp"],
            spec.row_group,
        )
        found = []
        for name, tokens in zip(rows["audio_filename"], rows["segment_timestamp"], strict=True):
            n, duration = chunk_index(name), last_timestamp(tokens)
            if n >= len(index) or name not in self._zip_of or duration is None:
                continue
            video = video_id(index[n].audio_filename)
            if video not in excluded:
                found.append(Candidate(name, strip_whisper_tokens(tokens), video, duration))
        log.info("mesolitica_context: %d candidates from %d index lines", len(found), len(index))
        return found

    def _indonesian_videos(self, index: Sequence[IndexLine]) -> set[str]:
        """Videos whose Malay-decoded transcript reads as Indonesian (see ``data.language``)."""
        texts: dict[str, list[str]] = defaultdict(list)
        for line in index:
            if whisper_language_code(line.text) == "ms":
                texts[video_id(line.audio_filename)].append(strip_whisper_tokens(line.text))
        spec = self._spec
        flagged = flag_indonesian(texts, spec.indonesian_max_share, spec.indonesian_min_hits)
        log.info(
            "Excluding %d of %d videos as Indonesian: %s", len(flagged), len(texts), sorted(flagged)
        )
        return flagged

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Read the chosen clips out of their remote zips."""
        by_zip: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            by_zip[self._zip_of[key]].append(key)
        for zip_name, names in by_zip.items():
            yield from _zip_members(self._fs, _hub_path(self._spec.repo, zip_name), names)


class MesoliticaConversational:
    """Malay Conversational Speech Corpus clips, grouped by the original corpus speaker id."""

    def __init__(self, spec: MesoliticaConversationalSource, fs: "HfFileSystem") -> None:
        self._spec = spec
        self._fs = fs

    def candidates(self) -> list[Candidate]:
        """Rows under the corpus prefix, joined to speaker ids by original row number.

        Durations come from the WAV sizes: this split's timestamps were re-aligned to the
        speech, so the last one can be far shorter than the clip.
        """
        spec = self._spec
        rows = _read_columns(
            self._fs, _hub_path(spec.repo, spec.parquet), ["audio_filename", "segment_timestamp"]
        )
        speakers = _read_columns(
            self._fs, _hub_path(spec.speakers_repo, spec.speakers_parquet), ["id"]
        )["id"]
        durations = _zip_wav_durations(self._fs, _hub_path(spec.repo, spec.zip), spec.prefix)
        found = []
        for name, tokens in zip(rows["audio_filename"], rows["segment_timestamp"], strict=True):
            duration = durations.get(name)
            if not name.startswith(spec.prefix) or duration is None:
                continue
            speaker = speakers[conversational_row(name, spec.prefix)]
            found.append(Candidate(name, strip_whisper_tokens(tokens), speaker, duration))
        log.info("malay_conversational: %d candidates", len(found))
        return found

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Read the chosen clips out of the remote zip."""
        yield from _zip_members(self._fs, _hub_path(self._spec.repo, self._spec.zip), keys)


class Fleurs:
    """One FLEURS split. There are no speaker ids, so a sentence (read by several people) is
    the group; held-out evaluation relies on the official split instead."""

    SAMPLE_RATE = 16000  # FLEURS num_samples are counted at 16 kHz

    def __init__(self, spec: FleursSource, fs: "HfFileSystem") -> None:
        self._spec = spec
        self._fs = fs
        split_dir = _hub_path(spec.repo, f"{spec.config}/{spec.split}", spec.revision)
        listing = fs.ls(split_dir, detail=False)
        self._files = sorted(p for p in listing if isinstance(p, str) and p.endswith(".parquet"))

    def candidates(self) -> list[Candidate]:
        """Every row, with duration taken from ``num_samples``."""
        found = []
        for path in self._files:
            rows = _read_columns(self._fs, path, ["id", "path", "raw_transcription", "num_samples"])
            for sid, key, text, samples in zip(
                rows["id"],
                rows["path"],
                rows["raw_transcription"],
                rows["num_samples"],
                strict=True,
            ):
                found.append(Candidate(key, text, f"sentence-{sid}", samples / self.SAMPLE_RATE))
        log.info("fleurs %s/%s: %d candidates", self._spec.config, self._spec.split, len(found))
        return found

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Audio column of only the row groups that hold the chosen rows."""
        for path in self._files:
            yield from _parquet_audio(self._fs, path, "path", keys)


class LibriSpeech:
    """Selected row groups of a LibriSpeech parquet; audio is embedded, so it is read once."""

    def __init__(self, spec: LibriSpeechSource, fs: "HfFileSystem") -> None:
        self._spec = spec
        self._fs = fs
        self._audio: dict[str, bytes] = {}

    def candidates(self) -> list[Candidate]:
        """Rows of the configured row groups, grouped by speaker, with exact durations."""
        found = []
        path = _hub_path(self._spec.repo, self._spec.parquet)
        for group in self._spec.row_groups:
            rows = _read_columns(self._fs, path, ["id", "text", "speaker_id", "audio"], group)
            for key, text, speaker, audio in zip(
                rows["id"], rows["text"], rows["speaker_id"], rows["audio"], strict=True
            ):
                self._audio[key] = audio["bytes"]
                duration = sf.info(io.BytesIO(audio["bytes"])).duration
                found.append(Candidate(key, text, f"speaker-{speaker}", duration))
        log.info("librispeech: %d candidates", len(found))
        return found

    def fetch(self, keys: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """Audio already read by ``candidates``."""
        for key in keys:
            yield key, self._audio[key]


def make_source(spec: SourceSpec, fs: "HfFileSystem | None" = None) -> ClipSource:
    """Build the source for a config entry."""
    hub = fs or _hub()
    match spec:
        case MesoliticaContextSource():
            return MesoliticaContext(spec, hub)
        case MesoliticaConversationalSource():
            return MesoliticaConversational(spec, hub)
        case FleursSource():
            return Fleurs(spec, hub)
        case LibriSpeechSource():
            return LibriSpeech(spec, hub)
