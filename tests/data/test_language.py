"""Tests for the Indonesian-vs-Malaysian video filter."""

from asr_assess.data.language import MarkerCounts, count_markers, flag_indonesian

INDONESIAN = "Udah nggak ada korban. Saya bilang saya siap bayar, mobilnya ada karena uang cukup."
MALAYSIAN = "Tak boleh lah macam tu, sebab kereta dia rosak. Korang nak pergi hospital ke?"


class TestCountMarkers:
    def test_indonesian_speech(self) -> None:
        counts = count_markers(INDONESIAN)
        assert counts.indonesian >= 5
        assert counts.malaysian == 0

    def test_malaysian_speech(self) -> None:
        counts = count_markers(MALAYSIAN)
        assert counts.malaysian >= 5
        assert counts.indonesian == 0

    def test_words_shared_by_formal_malay_are_not_evidence(self) -> None:
        # "ingin" and "tinggal" are standard Malaysian Malay too.
        assert count_markers("Saya ingin ucapkan tahniah, tinggal sedikit lagi") == MarkerCounts(
            0, 0
        )

    def test_is_case_insensitive_and_whole_word(self) -> None:
        assert count_markers("KARENA").indonesian == 1
        assert count_markers("karenanya ajaib").indonesian == 0


class TestFlagIndonesian:
    def test_flags_mostly_indonesian_video(self) -> None:
        texts = {"id-video": [INDONESIAN] * 3, "my-video": [MALAYSIAN] * 3}
        assert flag_indonesian(texts, max_share=0.2, min_hits=10) == {"id-video"}

    def test_keeps_video_with_occasional_indonesian_word(self) -> None:
        texts = {"v": [MALAYSIAN] * 5 + ["bisa juga"]}
        assert flag_indonesian(texts, max_share=0.2, min_hits=10) == set()

    def test_ignores_videos_with_too_little_evidence(self) -> None:
        assert flag_indonesian({"v": ["karena uang"]}, max_share=0.2, min_hits=10) == set()
