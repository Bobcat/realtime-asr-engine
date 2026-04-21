from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class PacingSettings:
    base_emit_ms: int
    startup_duration_ms: int = 0
    startup_emit_ms: int = 1
    startup_min_infer_audio_ms: int = 0
    startup_min_new_audio_ms: int = 0

    def normalized(self) -> "PacingSettings":
        return PacingSettings(
            base_emit_ms=int(max(1, int(self.base_emit_ms))),
            startup_duration_ms=int(max(0, int(self.startup_duration_ms))),
            startup_emit_ms=int(max(1, int(self.startup_emit_ms))),
            startup_min_infer_audio_ms=int(max(0, int(self.startup_min_infer_audio_ms))),
            startup_min_new_audio_ms=int(max(0, int(self.startup_min_new_audio_ms))),
        )


@dataclass(frozen=True)
class LivePacingSettings:
    enabled: bool = True
    min_emit_interval_ms: int = 0
    policy: PacingSettings = field(default_factory=lambda: PacingSettings(base_emit_ms=1))

    def normalized(self) -> "LivePacingSettings":
        return LivePacingSettings(
            enabled=bool(self.enabled),
            min_emit_interval_ms=int(max(0, int(self.min_emit_interval_ms))),
            policy=self.policy.normalized(),
        )


@dataclass(frozen=True)
class SileroVadSettings:
    enabled: bool = False
    venv: str | None = None
    threshold: float = 0.5
    max_speech_duration_s: float = 30.0
    min_speech_ms: int = 0
    hangover_ms: int = 0

    def normalized(self) -> "SileroVadSettings":
        return SileroVadSettings(
            enabled=bool(self.enabled),
            venv=(str(self.venv) if self.venv else None),
            threshold=float(max(0.0, min(1.0, float(self.threshold)))),
            max_speech_duration_s=float(max(0.1, float(self.max_speech_duration_s))),
            min_speech_ms=int(max(0, int(self.min_speech_ms))),
            hangover_ms=int(max(0, int(self.hangover_ms))),
        )


@dataclass(frozen=True)
class SpeechGateSettings:
    silence_enter_ms: int
    rearm_hits: int
    rearm_window_ms: int
    force_commit_silence_ms: int

    def normalized(self) -> "SpeechGateSettings":
        return SpeechGateSettings(
            silence_enter_ms=int(max(100, int(self.silence_enter_ms))),
            rearm_hits=int(max(1, int(self.rearm_hits))),
            rearm_window_ms=int(max(100, int(self.rearm_window_ms))),
            force_commit_silence_ms=int(max(100, int(self.force_commit_silence_ms))),
        )


@dataclass(frozen=True)
class RollingASRSettings:
    min_infer_audio_ms: int
    single_segment_commit_min_ms: int
    force_commit_repeats: int
    max_decode_window_ms: int
    max_uncommitted_ms: int
    hard_clip_keep_tail_ms: int
    buffer_trim_threshold_ms: int
    buffer_trim_drop_ms: int
    min_new_audio_ms: int = 0

    def normalized(self) -> "RollingASRSettings":
        min_infer_audio_ms = int(max(1, int(self.min_infer_audio_ms)))
        single_segment_commit_min_ms = int(max(min_infer_audio_ms, int(self.single_segment_commit_min_ms)))
        force_commit_repeats = int(max(1, int(self.force_commit_repeats)))
        max_decode_window_ms = int(max(min_infer_audio_ms, int(self.max_decode_window_ms)))
        max_uncommitted_ms = int(max(min_infer_audio_ms, int(self.max_uncommitted_ms)))
        if max_uncommitted_ms <= max_decode_window_ms:
            max_uncommitted_ms = int(max_decode_window_ms + min_infer_audio_ms)
        hard_clip_keep_tail_ms = int(
            max(
                min_infer_audio_ms,
                int(self.hard_clip_keep_tail_ms),
                single_segment_commit_min_ms,
            )
        )
        buffer_trim_threshold_ms = int(max(max_decode_window_ms, int(self.buffer_trim_threshold_ms)))
        buffer_trim_drop_ms = int(max(min_infer_audio_ms, int(self.buffer_trim_drop_ms)))
        min_new_audio_ms = int(max(0, int(self.min_new_audio_ms)))

        return RollingASRSettings(
            min_infer_audio_ms=min_infer_audio_ms,
            single_segment_commit_min_ms=single_segment_commit_min_ms,
            force_commit_repeats=force_commit_repeats,
            max_decode_window_ms=max_decode_window_ms,
            max_uncommitted_ms=max_uncommitted_ms,
            hard_clip_keep_tail_ms=hard_clip_keep_tail_ms,
            buffer_trim_threshold_ms=buffer_trim_threshold_ms,
            buffer_trim_drop_ms=buffer_trim_drop_ms,
            min_new_audio_ms=min_new_audio_ms,
        )


@dataclass(frozen=True)
class LiveASRRunnerSettings:
    rolling: RollingASRSettings
    pacing: LivePacingSettings = field(default_factory=LivePacingSettings)
    vad: SileroVadSettings = field(default_factory=SileroVadSettings)
    speech_gate: SpeechGateSettings = field(
        default_factory=lambda: SpeechGateSettings(
            silence_enter_ms=1000,
            rearm_hits=1,
            rearm_window_ms=1000,
            force_commit_silence_ms=1000,
        )
    )

    def normalized(self) -> "LiveASRRunnerSettings":
        return LiveASRRunnerSettings(
            rolling=self.rolling.normalized(),
            pacing=self.pacing.normalized(),
            vad=self.vad.normalized(),
            speech_gate=self.speech_gate.normalized(),
        )

    @classmethod
    def from_live_config(cls, live_config: Mapping[str, Any] | None) -> "LiveASRRunnerSettings":
        live_payload = _as_mapping(live_config)
        rolling_payload = _as_mapping(live_payload.get("rolling"))
        timing_payload = _as_mapping(live_payload.get("timing"))
        pacing_payload = _as_mapping(rolling_payload.get("pacing"))
        startup_payload = _as_mapping(pacing_payload.get("startup"))
        vad_payload = _as_mapping(rolling_payload.get("vad"))
        speech_gate_payload = _as_mapping(rolling_payload.get("speech_gate"))

        min_infer_audio_ms = _as_int(rolling_payload.get("min_infer_audio_ms"), 1000)
        min_new_audio_ms = _as_int(rolling_payload.get("min_new_audio_ms"), min_infer_audio_ms)
        pacing_base_emit_ms = _as_int(pacing_payload.get("base_emit_ms"), 500)

        return cls(
            rolling=RollingASRSettings(
                min_infer_audio_ms=min_infer_audio_ms,
                single_segment_commit_min_ms=_as_int(rolling_payload.get("single_segment_commit_min_ms"), 12000),
                force_commit_repeats=_as_int(rolling_payload.get("force_commit_repeats"), 8),
                max_decode_window_ms=_as_int(rolling_payload.get("max_decode_window_ms"), 12000),
                max_uncommitted_ms=_as_int(rolling_payload.get("max_uncommitted_ms"), 15000),
                hard_clip_keep_tail_ms=_as_int(rolling_payload.get("hard_clip_keep_tail_ms"), 5000),
                buffer_trim_threshold_ms=_as_int(rolling_payload.get("buffer_trim_threshold_ms"), 30000),
                buffer_trim_drop_ms=_as_int(rolling_payload.get("buffer_trim_drop_ms"), 20000),
                min_new_audio_ms=min_new_audio_ms,
            ),
            pacing=LivePacingSettings(
                enabled=_as_bool(pacing_payload.get("enabled"), True),
                min_emit_interval_ms=_as_int(timing_payload.get("emit_min_ms"), 250),
                policy=PacingSettings(
                    base_emit_ms=pacing_base_emit_ms,
                    startup_duration_ms=_as_int(startup_payload.get("duration_ms"), 0),
                    startup_emit_ms=_as_int(startup_payload.get("emit_ms"), pacing_base_emit_ms),
                    startup_min_infer_audio_ms=_as_int(startup_payload.get("min_infer_audio_ms"), min_infer_audio_ms),
                    startup_min_new_audio_ms=_as_int(startup_payload.get("min_new_audio_ms"), min_new_audio_ms),
                ),
            ),
            vad=SileroVadSettings(
                enabled=_as_bool(vad_payload.get("enabled"), False),
                venv=_as_optional_text(vad_payload.get("venv")),
                threshold=_as_float(vad_payload.get("threshold"), 0.35),
                max_speech_duration_s=_as_float(vad_payload.get("max_speech_duration_s"), 12.0),
                min_speech_ms=_as_int(vad_payload.get("min_speech_ms"), 120),
                hangover_ms=_as_int(vad_payload.get("hangover_ms"), 600),
            ),
            speech_gate=SpeechGateSettings(
                silence_enter_ms=_as_int(speech_gate_payload.get("silence_enter_ms"), 900),
                rearm_hits=_as_int(speech_gate_payload.get("rearm_hits"), 2),
                rearm_window_ms=_as_int(speech_gate_payload.get("rearm_window_ms"), 500),
                force_commit_silence_ms=_as_int(speech_gate_payload.get("force_commit_silence_ms"), 1500),
            ),
        ).normalized()
