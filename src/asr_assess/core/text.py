"""Transcript text handling: Whisper-token stripping, the WER normalizer, Qwen3-ASR prefixes."""

import re
import unicodedata

# Language names accepted by Qwen3-ASR, from qwen_asr/inference/utils.py (SUPPORTED_LANGUAGES).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "Chinese", "English", "Cantonese", "Arabic", "German", "French",
        "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
        "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
        "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
        "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
    }
)  # fmt: skip

ASR_TEXT_TAG = "<asr_text>"
LANGUAGE_PREFIX = "language "
NO_LANGUAGE = "None"

_WHISPER_TOKEN = re.compile(r"<\|[^|>]*\|>")
# A language token is a bare 2-3 letter code (e.g. <|ms|>, <|yue|>); tasks and timestamps are not.
_WHISPER_LANGUAGE = re.compile(r"<\|([a-z]{2,3})\|>")
_APOSTROPHES = re.compile(r"['’`]")
_NON_WORD = re.compile(r"[\W_]+")
_WHITESPACE = re.compile(r"\s+")


def strip_whisper_tokens(raw: str) -> str:
    """Remove Whisper special and timestamp tokens, leaving single-spaced plain text."""
    return _collapse(_WHISPER_TOKEN.sub(" ", raw))


def whisper_language_code(raw: str) -> str | None:
    """Return the Whisper language code (e.g. ``"ms"``) in a token string, if any."""
    match = _WHISPER_LANGUAGE.search(raw)
    return match.group(1) if match else None


def normalize_text(text: str) -> str:
    """Apply the fixed WER/CER normalizer: NFKC, lowercase, no punctuation, single spaces."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return _collapse(_NON_WORD.sub(" ", _APOSTROPHES.sub("", folded)))


def build_qwen_target(language: str | None, text: str) -> str:
    """Build a Qwen3-ASR training target such as ``language Malay<asr_text>...``."""
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported Qwen3-ASR language: {language!r}")
    return f"{LANGUAGE_PREFIX}{language or NO_LANGUAGE}{ASR_TEXT_TAG}{text}"


def parse_qwen_output(raw: str) -> tuple[str | None, str]:
    """Split Qwen3-ASR output into (language or None, transcript)."""
    if ASR_TEXT_TAG not in raw:
        return None, raw.strip()
    meta, text = raw.split(ASR_TEXT_TAG, 1)
    meta = meta.strip()
    if not meta.startswith(LANGUAGE_PREFIX):
        return None, text.strip()
    language = meta.removeprefix(LANGUAGE_PREFIX).strip()
    return (language if language and language != NO_LANGUAGE else None), text.strip()


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
