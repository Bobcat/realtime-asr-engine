from __future__ import annotations

import unittest

from realtime_asr_engine import PreviewHistoryState
from realtime_asr_engine import ResultInterpretationSettings
from realtime_asr_engine import TranscriptSegment
from realtime_asr_engine import interpret_asr_result
from realtime_asr_engine import reset_preview_history


class InterpretationTest(unittest.TestCase):
    def test_multiple_segments_commit_all_but_last(self) -> None:
        settings = ResultInterpretationSettings(
            single_segment_commit_min_ms=1500,
            force_commit_repeats=3,
        )
        segments = [
            TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=800),
            TranscriptSegment(segment_id="s2", text="world", t0_ms=800, t1_ms=1400),
        ]

        result = interpret_asr_result(
            t0_ms=0,
            t1_ms=1500,
            segments=segments,
            fallback_text="",
            settings=settings,
        )

        self.assertEqual(result.outcome, "commit")
        self.assertEqual(result.committed_segments, (segments[0],))
        self.assertEqual(result.preview_text, "world")
        self.assertEqual(result.preview_audio_end_ms, 1400)

    def test_single_short_segment_stays_preview(self) -> None:
        settings = ResultInterpretationSettings(
            single_segment_commit_min_ms=1500,
            force_commit_repeats=3,
        )
        segment = TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=900)

        result = interpret_asr_result(
            t0_ms=0,
            t1_ms=900,
            segments=[segment],
            fallback_text="",
            settings=settings,
        )

        self.assertEqual(result.outcome, "preview_only")
        self.assertEqual(result.committed_segments, ())
        self.assertEqual(result.preview_text, "hello")
        self.assertFalse(result.single_segment_forced_commit)

    def test_single_long_segment_forces_commit(self) -> None:
        settings = ResultInterpretationSettings(
            single_segment_commit_min_ms=1500,
            force_commit_repeats=3,
        )
        segment = TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=1700)

        result = interpret_asr_result(
            t0_ms=0,
            t1_ms=1700,
            segments=[segment],
            fallback_text="",
            settings=settings,
        )

        self.assertEqual(result.outcome, "commit")
        self.assertEqual(result.committed_segments, (segment,))
        self.assertEqual(result.preview_text, "")
        self.assertTrue(result.single_segment_forced_commit)

    def test_repeated_same_preview_audio_forces_commit(self) -> None:
        settings = ResultInterpretationSettings(
            single_segment_commit_min_ms=5000,
            force_commit_repeats=2,
        )
        segment = TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=1000)
        history = PreviewHistoryState(
            last_signature="hello",
            same_signature_repeats=1,
            last_audio_end_ms=1000,
            same_audio_end_repeats=1,
            last_preview_text="hello",
            last_preview_audio_end_fallback_ms=1000,
            last_preview_source_t0_ms=0,
        )

        result = interpret_asr_result(
            t0_ms=0,
            t1_ms=1000,
            segments=[segment],
            fallback_text="",
            settings=settings,
            preview_history=history,
        )

        self.assertEqual(result.outcome, "commit")
        self.assertEqual(result.committed_segments, (segment,))
        self.assertEqual(result.preview_text, "")
        self.assertTrue(result.force_commit_repeats_applied)
        self.assertTrue(result.reset_preview_history_on_commit)
        self.assertEqual(result.commit_reason, "rolling_context_force_commit_repeats")

    def test_reset_preview_history_returns_clean_state(self) -> None:
        self.assertEqual(reset_preview_history(), PreviewHistoryState())


if __name__ == "__main__":
    unittest.main()
