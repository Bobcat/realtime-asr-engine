from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from realtime_asr_engine import AudioFormat
from realtime_asr_engine import LiveASRRunner
from realtime_asr_engine import LiveASRRunnerSettings
from realtime_asr_engine import RollingASRSettings
from realtime_asr_engine import TranscriptSegment
from realtime_asr_engine.types import ASRResult
from realtime_asr_engine.types import ASRWorkItem


_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "minimal_host.py"
_EXAMPLE_SPEC = importlib.util.spec_from_file_location("realtime_asr_engine_example_minimal_host", _EXAMPLE_PATH)
if _EXAMPLE_SPEC is None or _EXAMPLE_SPEC.loader is None:
    raise RuntimeError(f"unable_to_load_example:{_EXAMPLE_PATH}")
_EXAMPLE_MODULE = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE_MODULE)
run_minimal_host_loop = _EXAMPLE_MODULE.run_minimal_host_loop


def _pcm_bytes_for_ms(ms: int, *, audio_format: AudioFormat) -> bytes:
    return b"\x00\x00" * (audio_format.ms_to_byte_offset(ms) // 2)


class _FakeAudioSource:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    async def next_frame(self) -> bytes | None:
        if self._frames:
            return self._frames.pop(0)
        return None


class _FakeExecutor:
    def __init__(self) -> None:
        self._pending: list[ASRWorkItem] = []

    async def submit(self, work_item: ASRWorkItem) -> None:
        self._pending.append(work_item)

    async def poll_completed(self) -> list[ASRResult]:
        if not self._pending:
            return []
        work_item = self._pending.pop(0)
        return [
            ASRResult(
                sequence_id=work_item.sequence_id,
                t0_ms=work_item.t0_ms,
                t1_ms=work_item.t1_ms,
                ok=True,
                segments=(
                    TranscriptSegment(
                        segment_id="seg-1",
                        text="hello world",
                        t0_ms=work_item.t0_ms,
                        t1_ms=work_item.t1_ms,
                    ),
                ),
            )
        ]


class _RecordingConsumer:
    def __init__(self) -> None:
        self.snapshots: list[tuple[int, str]] = []

    async def publish(self, transcript_state) -> None:
        committed_text = " ".join(seg.text for seg in transcript_state.committed_segments).strip()
        self.snapshots.append((int(transcript_state.transcript_revision), committed_text))


class MinimalHostTest(unittest.IsolatedAsyncioTestCase):
    async def test_minimal_host_loop_runs_audio_to_committed_transcript(self) -> None:
        audio_format = AudioFormat(sample_rate_hz=16000, channels=1, sample_width_bytes=2)
        runner = LiveASRRunner(
            audio_format=audio_format,
            settings=LiveASRRunnerSettings(
                rolling=RollingASRSettings(
                    min_infer_audio_ms=200,
                    single_segment_commit_min_ms=200,
                    force_commit_repeats=2,
                    max_decode_window_ms=4000,
                    max_uncommitted_ms=8000,
                    hard_clip_keep_tail_ms=2000,
                    buffer_trim_threshold_ms=3000,
                    buffer_trim_drop_ms=1000,
                    min_new_audio_ms=0,
                ),
            ),
        )
        source = _FakeAudioSource([_pcm_bytes_for_ms(300, audio_format=audio_format)])
        executor = _FakeExecutor()
        consumer = _RecordingConsumer()

        transcript_state = await run_minimal_host_loop(
            runner=runner,
            audio_source=source,
            asr_executor=executor,
            transcript_consumer=consumer,
            idle_sleep_s=0.0,
        )

        self.assertTrue(runner.is_drained())
        self.assertEqual(transcript_state.transcript_revision, 1)
        self.assertEqual(len(transcript_state.committed_segments), 1)
        self.assertEqual(transcript_state.committed_segments[0].text, "hello world")
        self.assertIn((1, "hello world"), consumer.snapshots)


if __name__ == "__main__":
    unittest.main()
