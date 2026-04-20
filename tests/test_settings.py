from __future__ import annotations

import unittest

from realtime_asr_engine import LiveASRRunnerSettings
from realtime_asr_engine import LivePacingSettings
from realtime_asr_engine import PacingSettings
from realtime_asr_engine import RollingASRSettings
from realtime_asr_engine import SileroVadSettings
from realtime_asr_engine import SpeechGateSettings


class SettingsTest(unittest.TestCase):
    def test_live_runner_settings_can_build_from_live_config_mapping(self) -> None:
        settings = LiveASRRunnerSettings.from_live_config(
            {
                "timing": {
                    "emit_min_ms": 120,
                },
                "rolling": {
                    "min_infer_audio_ms": 500,
                    "single_segment_commit_min_ms": 12000,
                    "force_commit_repeats": 3,
                    "max_uncommitted_ms": 30000,
                    "hard_clip_keep_tail_ms": 5000,
                    "max_decode_window_ms": 12000,
                    "buffer_trim_threshold_ms": 30000,
                    "buffer_trim_drop_ms": 20000,
                    "min_new_audio_ms": 500,
                    "pacing": {
                        "base_emit_ms": 250,
                        "startup": {
                            "duration_ms": 1200,
                            "emit_ms": 100,
                            "min_infer_audio_ms": 250,
                            "min_new_audio_ms": 200,
                        },
                    },
                    "vad": {
                        "enabled": True,
                        "whisperx_venv": "/fake/venv",
                        "threshold": 0.35,
                        "max_speech_duration_s": 12.0,
                        "min_speech_ms": 120,
                        "hangover_ms": 600,
                    },
                    "speech_gate": {
                        "silence_enter_ms": 900,
                        "rearm_hits": 2,
                        "rearm_window_ms": 500,
                        "force_commit_silence_ms": 2500,
                    },
                },
            }
        )

        self.assertEqual(settings.rolling.min_infer_audio_ms, 500)
        self.assertEqual(settings.rolling.force_commit_repeats, 3)
        self.assertEqual(settings.pacing.min_emit_interval_ms, 120)
        self.assertEqual(settings.pacing.policy.base_emit_ms, 250)
        self.assertEqual(settings.pacing.policy.startup_emit_ms, 100)
        self.assertTrue(settings.vad.enabled)
        self.assertEqual(settings.vad.whisperx_venv, "/fake/venv")
        self.assertEqual(settings.speech_gate.force_commit_silence_ms, 2500)

    def test_live_runner_settings_builder_uses_package_defaults(self) -> None:
        settings = LiveASRRunnerSettings.from_live_config({})

        self.assertEqual(settings.rolling.min_infer_audio_ms, 1000)
        self.assertEqual(settings.rolling.single_segment_commit_min_ms, 12000)
        self.assertEqual(settings.rolling.force_commit_repeats, 8)
        self.assertEqual(settings.rolling.min_new_audio_ms, 1000)
        self.assertEqual(settings.pacing.min_emit_interval_ms, 250)
        self.assertEqual(settings.pacing.policy.base_emit_ms, 500)
        self.assertFalse(settings.vad.enabled)
        self.assertEqual(settings.vad.threshold, 0.35)
        self.assertEqual(settings.speech_gate.silence_enter_ms, 900)
        self.assertEqual(settings.speech_gate.force_commit_silence_ms, 1500)

    def test_rolling_settings_normalize_cross_field_constraints(self) -> None:
        settings = RollingASRSettings(
            min_infer_audio_ms=800,
            single_segment_commit_min_ms=200,
            force_commit_repeats=0,
            max_decode_window_ms=500,
            max_uncommitted_ms=500,
            hard_clip_keep_tail_ms=300,
            buffer_trim_threshold_ms=400,
            buffer_trim_drop_ms=200,
            min_new_audio_ms=-10,
        ).normalized()

        self.assertEqual(settings.single_segment_commit_min_ms, 800)
        self.assertEqual(settings.force_commit_repeats, 1)
        self.assertEqual(settings.max_decode_window_ms, 800)
        self.assertEqual(settings.max_uncommitted_ms, 1600)
        self.assertEqual(settings.hard_clip_keep_tail_ms, 800)
        self.assertEqual(settings.buffer_trim_threshold_ms, 800)
        self.assertEqual(settings.buffer_trim_drop_ms, 800)
        self.assertEqual(settings.min_new_audio_ms, 0)

    def test_live_runner_settings_normalize_pacing_values(self) -> None:
        settings = LiveASRRunnerSettings(
            rolling=RollingASRSettings(
                min_infer_audio_ms=800,
                single_segment_commit_min_ms=200,
                force_commit_repeats=0,
                max_decode_window_ms=500,
                max_uncommitted_ms=500,
                hard_clip_keep_tail_ms=300,
                buffer_trim_threshold_ms=400,
                buffer_trim_drop_ms=200,
                min_new_audio_ms=-10,
            ),
            pacing=LivePacingSettings(
                enabled=True,
                min_emit_interval_ms=-20,
                policy=PacingSettings(base_emit_ms=0, startup_emit_ms=0),
            ),
            vad=SileroVadSettings(threshold=5.0, max_speech_duration_s=0.0, min_speech_ms=-1, hangover_ms=-1),
            speech_gate=SpeechGateSettings(
                silence_enter_ms=0,
                rearm_hits=0,
                rearm_window_ms=0,
                force_commit_silence_ms=0,
            ),
        ).normalized()

        self.assertTrue(settings.pacing.enabled)
        self.assertEqual(settings.pacing.min_emit_interval_ms, 0)
        self.assertEqual(settings.pacing.policy.base_emit_ms, 1)
        self.assertEqual(settings.pacing.policy.startup_emit_ms, 1)
        self.assertEqual(settings.vad.threshold, 1.0)
        self.assertEqual(settings.vad.max_speech_duration_s, 0.1)
        self.assertEqual(settings.vad.min_speech_ms, 0)
        self.assertEqual(settings.vad.hangover_ms, 0)
        self.assertEqual(settings.speech_gate.silence_enter_ms, 100)
        self.assertEqual(settings.speech_gate.rearm_hits, 1)
        self.assertEqual(settings.speech_gate.rearm_window_ms, 100)
        self.assertEqual(settings.speech_gate.force_commit_silence_ms, 100)


if __name__ == "__main__":
    unittest.main()
