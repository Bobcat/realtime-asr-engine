from __future__ import annotations

import unittest

from realtime_asr_engine import ASRResult
from realtime_asr_engine import AudioFormat
from realtime_asr_engine import RollingASRCore
from realtime_asr_engine import RollingASRSettings
from realtime_asr_engine import TranscriptSegment


def _pcm_bytes_for_ms(ms: int, *, audio_format: AudioFormat) -> bytes:
    return b"\x00\x00" * (audio_format.ms_to_byte_offset(ms) // 2)


class RollingASRCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_format = AudioFormat(sample_rate_hz=16000, channels=1, sample_width_bytes=2)
        self.settings = RollingASRSettings(
            min_infer_audio_ms=1000,
            single_segment_commit_min_ms=1500,
            force_commit_repeats=2,
            max_decode_window_ms=4000,
            max_uncommitted_ms=8000,
            hard_clip_keep_tail_ms=2000,
            buffer_trim_threshold_ms=3000,
            buffer_trim_drop_ms=1000,
        )

    def test_build_work_item_after_audio_ingest(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings, language="en")
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))

        decision = core.build_work_item()

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertIsNotNone(decision.work_item)
        self.assertEqual(decision.work_item.t0_ms, 0)
        self.assertEqual(decision.work_item.t1_ms, 1000)
        self.assertEqual(len(decision.work_item.pcm16le), self.audio_format.ms_to_byte_offset(1000))
        self.assertEqual(decision.work_item.language, "en")

    def test_second_work_item_is_blocked_while_inflight(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))

        first = core.build_work_item()
        second = core.build_work_item()

        self.assertEqual(first.reason, "work_item_ready")
        self.assertEqual(second.reason, "already_inflight")

    def test_finalize_input_allows_tail_work_below_threshold(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(200, audio_format=self.audio_format))

        before = core.build_work_item()
        core.finalize_input()
        after = core.build_work_item()

        self.assertEqual(before.reason, "insufficient_unprocessed_audio")
        self.assertEqual(after.reason, "work_item_ready")
        self.assertEqual(after.work_item.t1_ms, 200)

    def test_rollback_inflight_work_restores_last_submitted_offset(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))

        work = core.build_work_item().work_item
        rolled_back = core.rollback_inflight_work(sequence_id=work.sequence_id)

        self.assertTrue(rolled_back)
        self.assertFalse(core.has_inflight_work)
        self.assertEqual(core.last_submitted_t1_ms, 0)

    def test_clear_inflight_work_keeps_last_submitted_offset(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))

        work = core.build_work_item().work_item
        cleared = core.clear_inflight_work(sequence_id=work.sequence_id)

        self.assertTrue(cleared)
        self.assertFalse(core.has_inflight_work)
        self.assertEqual(core.last_submitted_t1_ms, 1000)

    def test_advance_offsets_to_moves_processed_and_decode_offsets(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(2000, audio_format=self.audio_format))

        core.advance_offsets_to(t1_ms=1200, update_last_submitted=True)

        self.assertEqual(core.processed_offset_ms, 1200)
        self.assertEqual(core.decode_offset_ms, 1200)
        self.assertEqual(core.last_submitted_t1_ms, 1200)

    def test_apply_preview_only_result_updates_preview_state(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        work = core.build_work_item().work_item

        decision = core.apply_result(
            ASRResult(
                sequence_id=work.sequence_id,
                t0_ms=work.t0_ms,
                t1_ms=work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=900),),
                text="hello",
            )
        )

        self.assertEqual(decision.reason, "preview_applied")
        self.assertEqual(core.transcript_state.transcript_revision, 0)
        self.assertEqual(core.transcript_state.preview.text, "hello")
        self.assertEqual(core.transcript_state.preview.audio_end_ms, 900)
        self.assertEqual(core.processed_offset_ms, 0)

    def test_apply_commit_result_advances_committed_state(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1700, audio_format=self.audio_format))
        work = core.build_work_item().work_item

        decision = core.apply_result(
            ASRResult(
                sequence_id=work.sequence_id,
                t0_ms=work.t0_ms,
                t1_ms=work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=1700),),
                text="hello",
            )
        )

        self.assertEqual(decision.reason, "commit_applied")
        self.assertEqual(core.transcript_state.transcript_revision, 1)
        self.assertEqual(len(core.transcript_state.committed_segments), 1)
        self.assertEqual(core.transcript_state.preview.text, "")
        self.assertEqual(core.processed_offset_ms, 1700)
        self.assertEqual(core.decode_offset_ms, 1700)

    def test_apply_result_can_commit_and_keep_trailing_preview(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1500, audio_format=self.audio_format))
        work = core.build_work_item().work_item

        decision = core.apply_result(
            ASRResult(
                sequence_id=work.sequence_id,
                t0_ms=work.t0_ms,
                t1_ms=work.t1_ms,
                ok=True,
                segments=(
                    TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=800),
                    TranscriptSegment(segment_id="s2", text="world", t0_ms=800, t1_ms=1400),
                ),
                text="hello world",
            )
        )

        self.assertEqual(decision.reason, "commit_applied")
        self.assertEqual(len(decision.committed_segments), 1)
        self.assertEqual(decision.preview.text, "world")
        self.assertEqual(core.transcript_state.preview.text, "world")
        self.assertEqual(core.transcript_state.preview.audio_end_ms, 1400)
        self.assertEqual(core.processed_offset_ms, 800)
        self.assertEqual(core.decode_offset_ms, 800)

    def test_commit_preview_tail_promotes_preview_to_committed(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1200, audio_format=self.audio_format))
        work = core.build_work_item(force=True).work_item
        core.apply_result(
            ASRResult(
                sequence_id=work.sequence_id,
                t0_ms=work.t0_ms,
                t1_ms=work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=900),),
                text="hello",
            )
        )

        committed = core.commit_preview_tail()

        self.assertIsNotNone(committed)
        self.assertEqual(committed.text, "hello")
        self.assertEqual(core.transcript_state.transcript_revision, 1)
        self.assertEqual(len(core.transcript_state.committed_segments), 1)
        self.assertEqual(core.transcript_state.committed_segments[0].text, "hello")
        self.assertEqual(core.transcript_state.committed_segments[0].t1_ms, 1200)
        self.assertEqual(core.transcript_state.preview.text, "")

    def test_manual_commit_preview_retires_overlapping_inflight_and_restarts_at_boundary(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        first_work = core.build_work_item(force=True).work_item
        core.apply_result(
            ASRResult(
                sequence_id=first_work.sequence_id,
                t0_ms=first_work.t0_ms,
                t1_ms=first_work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=900),),
                text="hello",
            )
        )
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        inflight_work = core.build_work_item(force=True).work_item

        decision = core.manual_commit_preview()

        self.assertTrue(decision.applied)
        self.assertEqual(decision.reason, "manual_preview_committed")
        self.assertEqual(decision.commit_reason, "manual_preview_commit")
        self.assertEqual(decision.retired_sequence_ids, (inflight_work.sequence_id,))
        self.assertEqual(decision.restart_t0_ms, 900)
        self.assertIsNotNone(decision.segment)
        self.assertEqual(decision.segment.text, "hello")
        self.assertEqual(decision.segment.t0_ms, 0)
        self.assertEqual(decision.segment.t1_ms, 900)
        self.assertFalse(core.has_inflight_work)
        self.assertEqual(core.processed_offset_ms, 900)
        self.assertEqual(core.decode_offset_ms, 900)
        self.assertEqual(core.last_submitted_t1_ms, 900)
        self.assertEqual(core.transcript_state.preview.text, "")

        late_decision = core.apply_result(
            ASRResult(
                sequence_id=inflight_work.sequence_id,
                t0_ms=inflight_work.t0_ms,
                t1_ms=inflight_work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s2", text="hello there", t0_ms=0, t1_ms=1900),),
                text="hello there",
            )
        )

        self.assertEqual(late_decision.reason, "retired_result")
        self.assertEqual(len(core.transcript_state.committed_segments), 1)
        self.assertEqual(core.transcript_state.committed_segments[0].text, "hello")
        self.assertEqual(core.transcript_state.preview.text, "")

        next_work = core.build_work_item(force=True).work_item

        self.assertIsNotNone(next_work)
        self.assertEqual(next_work.t0_ms, 900)
        self.assertEqual(next_work.t1_ms, 2000)

    def test_apply_commit_result_returns_committed_segments(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1700, audio_format=self.audio_format))
        work = core.build_work_item().work_item

        decision = core.apply_result(
            ASRResult(
                sequence_id=work.sequence_id,
                t0_ms=work.t0_ms,
                t1_ms=work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=1700),),
                text="hello",
            )
        )

        self.assertEqual(decision.reason, "commit_applied")
        self.assertEqual(len(decision.committed_segments), 1)
        self.assertEqual(decision.commit_reason, "rolling_context_commit")

    def test_full_commit_clears_stale_preview_tail_state(self) -> None:
        core = RollingASRCore(audio_format=self.audio_format, settings=self.settings)
        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        first_work = core.build_work_item(force=True).work_item

        first_decision = core.apply_result(
            ASRResult(
                sequence_id=first_work.sequence_id,
                t0_ms=first_work.t0_ms,
                t1_ms=first_work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s1", text="hello", t0_ms=0, t1_ms=900),),
                text="hello",
            )
        )

        self.assertEqual(first_decision.reason, "preview_applied")
        self.assertEqual(core.transcript_state.preview.text, "hello")

        core.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        second_work = core.build_work_item(force=True).work_item

        second_decision = core.apply_result(
            ASRResult(
                sequence_id=second_work.sequence_id,
                t0_ms=second_work.t0_ms,
                t1_ms=second_work.t1_ms,
                ok=True,
                segments=(TranscriptSegment(segment_id="s2", text="hello there", t0_ms=0, t1_ms=2000),),
                text="hello there",
            )
        )

        self.assertEqual(second_decision.reason, "commit_applied")
        self.assertEqual(core.transcript_state.preview.text, "")
        self.assertIsNone(core.commit_preview_tail(include_recording_end=False))

    def test_guardrail_metrics_update_after_hard_clip(self) -> None:
        settings = RollingASRSettings(
            min_infer_audio_ms=1000,
            single_segment_commit_min_ms=1500,
            force_commit_repeats=2,
            max_decode_window_ms=1000,
            max_uncommitted_ms=2000,
            hard_clip_keep_tail_ms=1000,
            buffer_trim_threshold_ms=3000,
            buffer_trim_drop_ms=1000,
        )
        core = RollingASRCore(audio_format=self.audio_format, settings=settings)
        core.ingest_audio(_pcm_bytes_for_ms(4000, audio_format=self.audio_format))

        decision = core.build_work_item()

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertEqual(core.guardrail_metrics["hard_clip_count"], 1)
        self.assertGreater(core.guardrail_metrics["hard_clip_dropped_audio_ms"], 0)


if __name__ == "__main__":
    unittest.main()
