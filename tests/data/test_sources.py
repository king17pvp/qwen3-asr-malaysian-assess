"""Tests for the pure parsing helpers behind the dataset sources."""

import json

import pytest

from asr_assess.data.sources import (
    IndexLine,
    chunk_index,
    conversational_row,
    last_timestamp,
    parse_index_head,
    pcm_wav_duration,
    video_id,
)


class TestMesoliticaNames:
    def test_chunk_index_is_the_index_line_number(self) -> None:
        assert chunk_index("prepared-pseudolabel-chunks/3037-0.mp3") == 3037

    def test_video_id_drops_the_window_number(self) -> None:
        # output-audio/<worker>-<video>-<30 s window>.mp3
        assert video_id("output-audio/2-0-2.mp3") == "2-0"

    def test_video_id_rejects_unexpected_names(self) -> None:
        with pytest.raises(ValueError):
            video_id("output-audio/weird.mp3")

    def test_conversational_row_is_the_wav_number(self) -> None:
        prefix = "malay-conversational-speech-corpus-whisper-format/"
        assert conversational_row(f"{prefix}12.wav", prefix) == 12


class TestLastTimestamp:
    def test_returns_the_final_timestamp(self) -> None:
        text = "<|startoftranscript|><|ms|><|transcribe|><|0.00|> a<|0.28|><|1.12|> b<|2.78|>"
        assert last_timestamp(text + "<|endoftext|>") == pytest.approx(2.78)

    def test_none_without_timestamps(self) -> None:
        assert last_timestamp("<|ms|> hello") is None


class TestParseIndexHead:
    def test_drops_the_trailing_partial_line(self) -> None:
        lines = [json.dumps({"new_text": f"t{i}", "audio_filename": f"output-audio/0-{i}-0.mp3"})
                 for i in range(3)]  # fmt: skip
        blob = ("\n".join(lines) + "\n" + lines[0][:10]).encode()
        assert parse_index_head(blob) == [
            IndexLine(f"output-audio/0-{i}-0.mp3", f"t{i}") for i in range(3)
        ]


class TestPcmWavDuration:
    def test_duration_from_byte_size(self) -> None:
        # 16 kHz mono 16-bit: 32000 bytes per second after the 44-byte header.
        assert pcm_wav_duration(44 + 64000, byte_rate=32000) == pytest.approx(2.0)

    def test_rejects_nonpositive_byte_rate(self) -> None:
        with pytest.raises(ValueError):
            pcm_wav_duration(100, byte_rate=0)
