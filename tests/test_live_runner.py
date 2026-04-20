from __future__ import annotations

import unittest
from unittest import mock

from realtime_asr_engine import AudioFormat
from realtime_asr_engine import ASRResult
from realtime_asr_engine import LiveASRRunner
from realtime_asr_engine import LiveASRRunnerSettings
from realtime_asr_engine import LivePacingSettings
from realtime_asr_engine import PacingSettings
from realtime_asr_engine import RollingASRSettings
from realtime_asr_engine import SileroVadSettings
from realtime_asr_engine import SpeechActivityObservation
from realtime_asr_engine import SpeechGateSettings
from realtime_asr_engine import TranscriptSegment


def _pcm_bytes_for_ms(ms: int, *, audio_format: AudioFormat) -> bytes:
    return b"\x00\x00" * (audio_format.ms_to_byte_offset(ms) // 2)


class LiveASRRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_format = AudioFormat(sample_rate_hz=16000, channels=1, sample_width_bytes=2)

    def test_startup_override_can_issue_work_before_base_threshold(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
            settings=LiveASRRunnerSettings(
                rolling=RollingASRSettings(
                    min_infer_audio_ms=1000,
                    single_segment_commit_min_ms=1500,
                    force_commit_repeats=2,
                    max_decode_window_ms=4000,
                    max_uncommitted_ms=8000,
                    hard_clip_keep_tail_ms=2000,
                    buffer_trim_threshold_ms=3000,
                    buffer_trim_drop_ms=1000,
                    min_new_audio_ms=0,
                ),
                pacing=LivePacingSettings(
                    enabled=True,
                    min_emit_interval_ms=0,
                    policy=PacingSettings(
                        base_emit_ms=250,
                        startup_duration_ms=1200,
                        startup_emit_ms=1,
                        startup_min_infer_audio_ms=200,
                        startup_min_new_audio_ms=0,
                    ),
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(200, audio_format=self.audio_format))

        decision = runner.build_work_item(now_mono=1.0)

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertIsNotNone(decision.work_item)
        self.assertEqual(decision.work_item.t1_ms, 200)

    def test_startup_emit_interval_can_block_fast_retry(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
            settings=LiveASRRunnerSettings(
                rolling=RollingASRSettings(
                    min_infer_audio_ms=1000,
                    single_segment_commit_min_ms=1500,
                    force_commit_repeats=2,
                    max_decode_window_ms=4000,
                    max_uncommitted_ms=8000,
                    hard_clip_keep_tail_ms=2000,
                    buffer_trim_threshold_ms=3000,
                    buffer_trim_drop_ms=1000,
                    min_new_audio_ms=0,
                ),
                pacing=LivePacingSettings(
                    enabled=True,
                    min_emit_interval_ms=0,
                    policy=PacingSettings(
                        base_emit_ms=250,
                        startup_duration_ms=1200,
                        startup_emit_ms=100,
                        startup_min_infer_audio_ms=200,
                        startup_min_new_audio_ms=0,
                    ),
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))
        first = runner.build_work_item(now_mono=1.0)
        runner.rollback_inflight_work(sequence_id=first.work_item.sequence_id)

        second = runner.build_work_item(now_mono=1.05)

        self.assertEqual(first.reason, "work_item_ready")
        self.assertEqual(second.reason, "emit_interval_wait")
        self.assertEqual(runner.guardrail_metrics["emit_interval_skips"], 1)

    def test_steady_state_pacing_can_block_same_slot_retry(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                pacing=LivePacingSettings(
                    enabled=True,
                    min_emit_interval_ms=0,
                    policy=PacingSettings(
                        base_emit_ms=250,
                        startup_duration_ms=0,
                    ),
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))
        first = runner.build_work_item(now_mono=1.0)
        runner.rollback_inflight_work(sequence_id=first.work_item.sequence_id)

        second = runner.build_work_item(now_mono=1.10)

        self.assertEqual(first.reason, "work_item_ready")
        self.assertEqual(second.reason, "pacing_slot_wait")
        self.assertEqual(runner.guardrail_metrics["pacing_slot_skips"], 1)

    def test_disabled_pacing_does_not_block_build(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                pacing=LivePacingSettings(
                    enabled=False,
                    min_emit_interval_ms=500,
                    policy=PacingSettings(
                        base_emit_ms=500,
                        startup_duration_ms=1000,
                        startup_emit_ms=500,
                        startup_min_infer_audio_ms=1000,
                        startup_min_new_audio_ms=1000,
                    ),
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

        decision = runner.build_work_item(now_mono=1.0)

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertIsNotNone(decision.work_item)

    def test_speech_gate_quiet_waits_for_rearm_hits(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
                speech_gate=SpeechGateSettings(
                    silence_enter_ms=900,
                    rearm_hits=2,
                    rearm_window_ms=500,
                    force_commit_silence_ms=2500,
                ),
            ),
        )

        first = runner.handle_speech_activity(
            now_mono=1.0,
            observation=SpeechActivityObservation(speech_hit=True, reason="speech"),
            rearm_from_ms=0,
        )
        second = runner.handle_speech_activity(
            now_mono=1.1,
            observation=SpeechActivityObservation(speech_hit=True, reason="speech"),
            rearm_from_ms=0,
        )

        self.assertEqual(first.reason, "quiet_waiting_rearm")
        self.assertFalse(first.allow_work)
        self.assertEqual(first.next_state, "quiet")
        self.assertEqual(second.reason, "speech_hit")
        self.assertTrue(second.allow_work)
        self.assertEqual(second.next_state, "active")
        self.assertEqual(runner.speech_gate_state, "active")
        self.assertEqual(runner.guardrail_metrics["speech_gate_rearm_count"], 1)

    def test_speech_gate_active_silence_requests_force_commit(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
                speech_gate=SpeechGateSettings(
                    silence_enter_ms=900,
                    rearm_hits=1,
                    rearm_window_ms=500,
                    force_commit_silence_ms=2500,
                ),
            ),
        )
        runner.handle_speech_activity(
            now_mono=1.0,
            observation=SpeechActivityObservation(speech_hit=True, reason="speech"),
            rearm_from_ms=0,
        )

        decision = runner.handle_speech_activity(
            now_mono=4.0,
            observation=SpeechActivityObservation(speech_hit=False, reason="silence"),
            rearm_from_ms=250,
        )

        self.assertFalse(decision.allow_work)
        self.assertTrue(decision.force_commit_requested)
        self.assertEqual(decision.previous_state, "active")
        self.assertEqual(decision.next_state, "quiet")
        self.assertEqual(runner.speech_gate_state, "quiet")
        self.assertEqual(runner.speech_gate_rearm_from_ms, 250)
        self.assertIsNone(
            runner.commit_preview_tail(
                include_recording_end=False,
                max_t1_ms=250,
                speech_gate_forced=True,
            )
        )
        self.assertEqual(runner.guardrail_metrics["speech_gate_forced_commit_count"], 0)

    def test_speech_gate_forced_commit_metric_increments_on_tail_commit(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
            settings=LiveASRRunnerSettings(
                rolling=RollingASRSettings(
                    min_infer_audio_ms=200,
                    single_segment_commit_min_ms=1500,
                    force_commit_repeats=2,
                    max_decode_window_ms=4000,
                    max_uncommitted_ms=8000,
                    hard_clip_keep_tail_ms=2000,
                    buffer_trim_threshold_ms=3000,
                    buffer_trim_drop_ms=1000,
                    min_new_audio_ms=0,
                ),
                vad=SileroVadSettings(enabled=True),
                speech_gate=SpeechGateSettings(
                    silence_enter_ms=900,
                    rearm_hits=1,
                    rearm_window_ms=500,
                    force_commit_silence_ms=2500,
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(1000, audio_format=self.audio_format))
        runner.handle_speech_activity(
            now_mono=1.0,
            observation=SpeechActivityObservation(speech_hit=True, reason="speech"),
            rearm_from_ms=0,
        )
        work = runner.build_work_item(now_mono=1.0)
        self.assertIsNotNone(work.work_item)
        runner.apply_result(
            ASRResult(
                sequence_id=work.work_item.sequence_id,
                t0_ms=work.work_item.t0_ms,
                t1_ms=work.work_item.t1_ms,
                ok=True,
                segments=(
                    TranscriptSegment(
                        segment_id="seg1",
                        text="hello world",
                        t0_ms=0,
                        t1_ms=1000,
                    ),
                ),
            )
        )

        decision = runner.handle_speech_activity(
            now_mono=4.0,
            observation=SpeechActivityObservation(speech_hit=False, reason="silence"),
            rearm_from_ms=work.work_item.t1_ms,
        )

        self.assertTrue(decision.force_commit_requested)
        committed = runner.commit_preview_tail(
            include_recording_end=False,
            max_t1_ms=work.work_item.t1_ms,
            speech_gate_forced=True,
        )

        self.assertIsNotNone(committed)
        self.assertEqual(runner.guardrail_metrics["speech_gate_forced_commit_count"], 1)
        self.assertEqual(len(runner.transcript_state.committed_segments), 1)

    def test_observe_speech_activity_counts_hangover_as_silence_for_guardrails(self) -> None:
        class HangoverDetector:
            def should_enqueue_pcm16(
                self,
                _pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                return {
                    "allow": True,
                    "reason": "hangover",
                    "speech_ms": 60,
                    "segments_count": 1,
                }

        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

        observation = runner.observe_speech_activity(
            detector=HangoverDetector(),
            now_mono=1.0,
            end_ms=300,
            pending_t0_ms=0,
        )

        self.assertFalse(observation.speech_hit)
        self.assertEqual(observation.reason, "hangover")
        self.assertEqual(runner.guardrail_metrics["vad_hangover_allows"], 0)
        self.assertEqual(runner.guardrail_metrics["vad_silence_skips"], 1)

    def test_engine_runtime_payload_uses_internal_vad_detector(self) -> None:
        class FakeGate:
            def __init__(self, *, settings, sample_rate_hz: int) -> None:
                self.settings = settings
                self.sample_rate_hz = int(sample_rate_hz)

            def config_payload(self) -> dict[str, object]:
                return {
                    "enabled": True,
                    "provider": "silero",
                    "sample_rate_hz": self.sample_rate_hz,
                    "threshold": float(self.settings.threshold),
                    "max_speech_duration_s": float(self.settings.max_speech_duration_s),
                    "min_speech_ms": int(self.settings.min_speech_ms),
                    "hangover_ms": int(self.settings.hangover_ms),
                    "whisperx_venv": str(self.settings.whisperx_venv or ""),
                    "site_packages": "/fake/site-packages",
                }

            def state_payload(self) -> dict[str, object]:
                return {
                    "checks": 7,
                    "speech_checks": 3,
                    "silence_checks": 4,
                    "hangover_allows": 1,
                    "last_speech_age_ms": 120,
                }

        with mock.patch("realtime_asr_engine.runner.SileroVadGate", FakeGate):
            runner = LiveASRRunner(
                audio_format=self.audio_format,
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
                    vad=SileroVadSettings(
                        enabled=True,
                        whisperx_venv="/fake/venv",
                        threshold=0.35,
                        max_speech_duration_s=12.0,
                        min_speech_ms=120,
                        hangover_ms=600,
                    ),
                ),
            )

            payload = runner.engine_runtime_payload(now_mono=1.0)

        self.assertTrue(payload["vad"]["enabled"])
        self.assertEqual(payload["vad"]["config"]["provider"], "silero")
        self.assertEqual(payload["vad"]["config"]["threshold"], 0.35)
        self.assertEqual(payload["vad"]["state"]["checks"], 7)
        self.assertEqual(payload["speech_gate"]["state"], "quiet")
        self.assertIn("debug", payload)
        self.assertEqual(payload["debug"]["state"]["recording_duration_ms"], 0)
        self.assertEqual(payload["debug"]["reason_counts"]["work_decision"], {})

    def test_debug_snapshot_tracks_work_apply_and_tail_commit_reasons(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
            settings=LiveASRRunnerSettings(
                rolling=RollingASRSettings(
                    min_infer_audio_ms=200,
                    single_segment_commit_min_ms=1000,
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
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))
        work = runner.build_work_item(now_mono=1.0)
        self.assertIsNotNone(work.work_item)

        apply = runner.apply_result(
            ASRResult(
                sequence_id=work.work_item.sequence_id,
                t0_ms=work.work_item.t0_ms,
                t1_ms=work.work_item.t1_ms,
                ok=True,
                segments=(
                    TranscriptSegment(
                        segment_id="seg1",
                        text="hello world",
                        t0_ms=0,
                        t1_ms=300,
                    ),
                ),
            )
        )
        self.assertEqual(apply.reason, "preview_applied")
        committed = runner.commit_preview_tail(include_recording_end=False, max_t1_ms=300)
        self.assertIsNotNone(committed)

        debug = runner.debug_snapshot()

        self.assertEqual(debug["reason_counts"]["work_decision"]["work_item_ready"], 1)
        self.assertEqual(debug["reason_counts"]["apply_decision"]["preview_applied"], 1)
        self.assertEqual(debug["reason_counts"]["commit_reason"]["rolling_context_tail_preview_commit"], 1)
        self.assertEqual(debug["state"]["processed_offset_ms"], 300)
        self.assertEqual(debug["state"]["unprocessed_audio_ms"], 0)
        self.assertEqual(debug["state"]["preview_chars"], 0)
        self.assertIsNone(debug["state"]["inflight"])

    def test_maybe_dispatch_work_uses_internal_vad_detector_when_no_override_is_passed(self) -> None:
        class FakeGate:
            calls: list[dict[str, object]] = []

            def __init__(self, *, settings, sample_rate_hz: int) -> None:
                self.settings = settings
                self.sample_rate_hz = int(sample_rate_hz)

            def should_enqueue_pcm16(
                self,
                pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                type(self).calls.append(
                    {
                        "bytes": len(pcm16le),
                        "now_mono": now_mono,
                        "allow_hangover": allow_hangover,
                    }
                )
                return {
                    "allow": True,
                    "reason": "speech",
                    "speech_ms": 120,
                    "segments_count": 1,
                }

            def config_payload(self) -> dict[str, object]:
                return {"provider": "silero"}

            def state_payload(self) -> dict[str, object]:
                return {}

        with mock.patch("realtime_asr_engine.runner.SileroVadGate", FakeGate):
            runner = LiveASRRunner(
                audio_format=self.audio_format,
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
                    vad=SileroVadSettings(enabled=True, whisperx_venv="/fake/venv"),
                    speech_gate=SpeechGateSettings(
                        silence_enter_ms=900,
                        rearm_hits=1,
                        rearm_window_ms=500,
                        force_commit_silence_ms=2500,
                    ),
                ),
            )
            runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

            decision = runner.maybe_dispatch_work(now_mono=1.0)

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertIsNotNone(decision.work_decision.work_item)
        self.assertEqual(len(FakeGate.calls), 1)
        self.assertFalse(FakeGate.calls[0]["allow_hangover"])
        self.assertEqual(runner.guardrail_metrics["vad_checks"], 1)
        self.assertEqual(runner.guardrail_metrics["vad_speech_allows"], 1)

    def test_disabled_vad_always_allows_work(self) -> None:
        runner = LiveASRRunner(
            audio_format=self.audio_format,
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

        decision = runner.handle_speech_activity(
            now_mono=1.0,
            observation=SpeechActivityObservation(speech_hit=False, reason="silence"),
            rearm_from_ms=0,
        )

        self.assertTrue(decision.allow_work)
        self.assertEqual(decision.reason, "vad_disabled")

    def test_observe_speech_activity_uses_detector_and_updates_metrics(self) -> None:
        class FakeDetector:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def should_enqueue_pcm16(
                self,
                pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                self.calls.append(
                    {
                        "bytes": len(pcm16le),
                        "now_mono": now_mono,
                        "allow_hangover": allow_hangover,
                    }
                )
                return {
                    "allow": True,
                    "reason": "speech",
                    "speech_ms": 120,
                    "segments_count": 2,
                }

        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
                speech_gate=SpeechGateSettings(
                    silence_enter_ms=900,
                    rearm_hits=1,
                    rearm_window_ms=500,
                    force_commit_silence_ms=2500,
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))
        detector = FakeDetector()

        observation = runner.observe_speech_activity(
            detector=detector,
            now_mono=1.0,
            end_ms=300,
            pending_t0_ms=0,
        )

        self.assertTrue(observation.speech_hit)
        self.assertEqual(observation.reason, "speech")
        self.assertEqual(observation.speech_ms, 120)
        self.assertEqual(observation.segments_count, 2)
        self.assertEqual(len(detector.calls), 1)
        self.assertFalse(detector.calls[0]["allow_hangover"])
        self.assertEqual(runner.guardrail_metrics["vad_checks"], 1)
        self.assertEqual(runner.guardrail_metrics["vad_speech_allows"], 1)

    def test_observe_speech_activity_reports_detector_error(self) -> None:
        class BrokenDetector:
            def should_enqueue_pcm16(
                self,
                _pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                raise RuntimeError("vad boom")

        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

        observation = runner.observe_speech_activity(
            detector=BrokenDetector(),
            now_mono=1.0,
            end_ms=300,
            pending_t0_ms=0,
        )

        self.assertFalse(observation.speech_hit)
        self.assertEqual(observation.reason, "vad_error")
        self.assertIn("RuntimeError", observation.error)
        self.assertEqual(runner.guardrail_metrics["vad_checks"], 1)
        self.assertEqual(runner.guardrail_metrics["vad_errors"], 1)

    def test_maybe_dispatch_work_returns_vad_error_without_work_item(self) -> None:
        class BrokenDetector:
            def should_enqueue_pcm16(
                self,
                _pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                raise RuntimeError("vad boom")

        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

        decision = runner.maybe_dispatch_work(
            now_mono=1.0,
            detector=BrokenDetector(),
        )

        self.assertEqual(decision.reason, "vad_error")
        self.assertEqual(decision.work_decision.reason, "vad_error")
        self.assertIsNone(decision.work_decision.work_item)
        self.assertIn("RuntimeError", decision.error)

    def test_maybe_dispatch_work_returns_ready_work_after_rearm(self) -> None:
        class SpeechDetector:
            def should_enqueue_pcm16(
                self,
                _pcm16le: bytes,
                *,
                now_mono: float | None = None,
                allow_hangover: bool = True,
            ) -> dict[str, object]:
                return {
                    "allow": True,
                    "reason": "speech",
                    "speech_ms": 120,
                    "segments_count": 1,
                }

        runner = LiveASRRunner(
            audio_format=self.audio_format,
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
                vad=SileroVadSettings(enabled=True),
                speech_gate=SpeechGateSettings(
                    silence_enter_ms=900,
                    rearm_hits=1,
                    rearm_window_ms=500,
                    force_commit_silence_ms=2500,
                ),
            ),
        )
        runner.ingest_audio(_pcm_bytes_for_ms(300, audio_format=self.audio_format))

        decision = runner.maybe_dispatch_work(
            now_mono=1.0,
            detector=SpeechDetector(),
        )

        self.assertEqual(decision.reason, "work_item_ready")
        self.assertIsNotNone(decision.work_decision.work_item)
        self.assertIsNotNone(decision.speech_observation)
        self.assertIsNotNone(decision.speech_gate_decision)
        self.assertEqual(decision.speech_gate_decision.reason, "speech_hit")


if __name__ == "__main__":
    unittest.main()
