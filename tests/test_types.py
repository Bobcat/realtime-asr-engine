from __future__ import annotations

import unittest

from realtime_asr_engine import ASRResult
from realtime_asr_engine import ASRWorkItem
from realtime_asr_engine import AudioFormat
from realtime_asr_engine import PreviewTranscriptState
from realtime_asr_engine import TranscriptSegment
from realtime_asr_engine import TranscriptState


class TypesTest(unittest.TestCase):
    def test_transcript_segment_round_trips_via_dict(self) -> None:
        segment = TranscriptSegment(
            segment_id="s0001",
            text="hello world",
            t0_ms=120,
            t1_ms=980,
            speaker="SPK_1",
        )

        restored = TranscriptSegment.from_dict(segment.to_dict())

        self.assertEqual(restored, segment)

    def test_transcript_state_defaults(self) -> None:
        state = TranscriptState()

        self.assertEqual(state.transcript_revision, 0)
        self.assertEqual(state.committed_segments, [])
        self.assertEqual(state.preview, PreviewTranscriptState())

    def test_asr_result_carries_segments(self) -> None:
        segment = TranscriptSegment(
            segment_id="s0001",
            text="hello world",
            t0_ms=0,
            t1_ms=1000,
        )
        work = ASRWorkItem(sequence_id=7, t0_ms=0, t1_ms=1000, language="en")
        result = ASRResult(
            sequence_id=work.sequence_id,
            t0_ms=work.t0_ms,
            t1_ms=work.t1_ms,
            segments=(segment,),
            text="hello world",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.segments, (segment,))

    def test_audio_format_converts_between_ms_and_bytes(self) -> None:
        audio_format = AudioFormat(sample_rate_hz=16000, channels=1, sample_width_bytes=2)

        self.assertEqual(audio_format.bytes_per_second, 32000)
        self.assertEqual(audio_format.ms_to_byte_offset(1000), 32000)
        self.assertEqual(audio_format.bytes_to_ms(32000), 1000)
        self.assertEqual(audio_format.ms_to_byte_offset(1), 32)


if __name__ == "__main__":
    unittest.main()
