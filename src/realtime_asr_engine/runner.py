from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any
from typing import Literal
from typing import Mapping
from typing import Protocol

from .audio import AudioFormat
from .core import RollingASRCore
from .settings import LiveASRRunnerSettings
from .types import ASRResult
from .types import ApplyDecision
from .types import PreviewCommitDecision
from .types import TranscriptState
from .types import WorkDecision
from .vad_silero import SileroVadGate


@dataclass
class _LiveRunnerRuntime:
    last_emit_mono: float = 0.0
    pacing_epoch_mono_ms: int = 0
    pacing_last_slot_index: int = -1
    emit_interval_skips: int = 0
    pacing_slot_skips: int = 0
    vad_checks: int = 0
    vad_speech_allows: int = 0
    vad_hangover_allows: int = 0
    vad_silence_skips: int = 0
    vad_errors: int = 0
    speech_gate_state: Literal["quiet", "active"] = "active"
    speech_gate_recent_hits_mono: list[float] = field(default_factory=list)
    last_recent_speech_mono: float = 0.0
    speech_gate_rearm_from_ms: int = 0
    speech_gate_state_transitions: int = 0
    speech_gate_rearm_count: int = 0
    speech_gate_quiet_skips: int = 0
    speech_gate_forced_commit_count: int = 0
    speech_gate_silence_flush_count: int = 0


@dataclass(frozen=True)
class SpeechActivityObservation:
    speech_hit: bool
    reason: str = ""
    speech_ms: int = 0
    segments_count: int = 0
    error: str = ""


class SpeechActivityDetector(Protocol):
    def should_enqueue_pcm16(
        self,
        pcm16le: bytes,
        *,
        now_mono: float | None = None,
        allow_hangover: bool = True,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class SpeechGateDecision:
    allow_work: bool
    reason: str
    force_commit_requested: bool = False
    previous_state: str = ""
    next_state: str = ""
    silence_elapsed_ms: int | None = None


@dataclass(frozen=True)
class DispatchDecision:
    reason: str
    work_decision: WorkDecision
    speech_observation: SpeechActivityObservation | None = None
    speech_gate_decision: SpeechGateDecision | None = None
    error: str = ""


class LiveASRRunner:
    def __init__(
        self,
        *,
        audio_format: AudioFormat,
        settings: LiveASRRunnerSettings,
        language: str | None = None,
    ) -> None:
        self.audio_format = audio_format
        self.settings = settings.normalized()
        self.core = RollingASRCore(
            audio_format=audio_format,
            settings=self.settings.rolling,
            language=language,
        )
        self._rt = _LiveRunnerRuntime()
        self._detector: SpeechActivityDetector | None = None
        if bool(self.settings.vad.enabled):
            self._rt.speech_gate_state = "quiet"

    @property
    def language(self) -> str | None:
        return self.core.language

    @language.setter
    def language(self, value: str | None) -> None:
        self.core.language = value

    @property
    def recording_duration_ms(self) -> int:
        return int(self.core.recording_duration_ms)

    @property
    def pcm_base_ms(self) -> int:
        return int(self.core.pcm_base_ms)

    @property
    def processed_offset_ms(self) -> int:
        return int(self.core.processed_offset_ms)

    @property
    def decode_offset_ms(self) -> int:
        return int(self.core.decode_offset_ms)

    @property
    def last_submitted_t1_ms(self) -> int:
        return int(self.core.last_submitted_t1_ms)

    @property
    def preview_history(self):
        return self.core.preview_history

    @property
    def transcript_state(self) -> TranscriptState:
        return self.core.transcript_state

    @property
    def pacing_last_slot_index(self) -> int:
        return int(self._rt.pacing_last_slot_index)

    @property
    def speech_gate_state(self) -> str:
        return str(self._rt.speech_gate_state)

    @property
    def speech_gate_recent_hits_count(self) -> int:
        return int(max(0, len(self._rt.speech_gate_recent_hits_mono)))

    @property
    def speech_gate_rearm_from_ms(self) -> int:
        return int(max(0, self._rt.speech_gate_rearm_from_ms))

    @property
    def last_recent_speech_mono(self) -> float:
        return float(max(0.0, self._rt.last_recent_speech_mono))

    @property
    def speech_gate_recent_hits_mono(self) -> list[float]:
        return list(self._rt.speech_gate_recent_hits_mono)

    @property
    def guardrail_metrics(self) -> dict[str, int]:
        metrics = dict(self.core.guardrail_metrics)
        metrics["emit_interval_skips"] = int(max(0, self._rt.emit_interval_skips))
        metrics["pacing_slot_skips"] = int(max(0, self._rt.pacing_slot_skips))
        metrics["vad_checks"] = int(max(0, self._rt.vad_checks))
        metrics["vad_speech_allows"] = int(max(0, self._rt.vad_speech_allows))
        metrics["vad_hangover_allows"] = int(max(0, self._rt.vad_hangover_allows))
        metrics["vad_silence_skips"] = int(max(0, self._rt.vad_silence_skips))
        metrics["vad_errors"] = int(max(0, self._rt.vad_errors))
        metrics["speech_gate_state_transitions"] = int(max(0, self._rt.speech_gate_state_transitions))
        metrics["speech_gate_rearm_count"] = int(max(0, self._rt.speech_gate_rearm_count))
        metrics["speech_gate_quiet_skips"] = int(max(0, self._rt.speech_gate_quiet_skips))
        metrics["speech_gate_forced_commit_count"] = int(max(0, self._rt.speech_gate_forced_commit_count))
        metrics["speech_gate_silence_flush_count"] = int(max(0, self._rt.speech_gate_silence_flush_count))
        return metrics

    def ensure_vad_ready(self) -> None:
        if (not bool(self.settings.vad.enabled)) or (self._detector is not None):
            return
        self._detector = SileroVadGate(
            settings=self.settings.vad,
            sample_rate_hz=int(self.audio_format.sample_rate_hz),
        )

    def debug_snapshot(self) -> dict[str, Any]:
        return self.core.debug_snapshot()

    def engine_runtime_payload(self, *, now_mono: float | None = None) -> dict[str, Any]:
        if bool(self.settings.vad.enabled):
            self.ensure_vad_ready()

        vad_payload: dict[str, Any]
        if self._detector is not None and hasattr(self._detector, "config_payload") and hasattr(self._detector, "state_payload"):
            vad_payload = {
                "enabled": True,
                "config": dict(self._detector.config_payload()),
                "state": dict(self._detector.state_payload()),
            }
        else:
            vad_payload = {
                "enabled": False,
                "config": {
                    "provider": "silero",
                    "threshold": float(self.settings.vad.threshold),
                    "max_speech_duration_s": float(self.settings.vad.max_speech_duration_s),
                    "min_speech_ms": int(self.settings.vad.min_speech_ms),
                    "hangover_ms": int(self.settings.vad.hangover_ms),
                    "venv": str(self.settings.vad.venv or ""),
                    "sample_rate_hz": int(self.audio_format.sample_rate_hz),
                },
                "state": {},
            }

        silence_elapsed_ms = None
        safe_now_mono = float(now_mono if now_mono is not None else time.monotonic())
        if float(self.last_recent_speech_mono) > 0.0:
            silence_elapsed_ms = int(max(0.0, safe_now_mono - float(self.last_recent_speech_mono)) * 1000.0)

        return {
            "vad": vad_payload,
            "speech_gate": {
                "state": str(self.speech_gate_state),
                "recent_hits_count": int(self.speech_gate_recent_hits_count),
                "silence_elapsed_ms": silence_elapsed_ms,
                "rearm_from_ms": int(self.speech_gate_rearm_from_ms),
                "silence_enter_ms": int(self.settings.speech_gate.silence_enter_ms),
                "rearm_hits": int(self.settings.speech_gate.rearm_hits),
                "rearm_window_ms": int(self.settings.speech_gate.rearm_window_ms),
                "force_commit_silence_ms": int(self.settings.speech_gate.force_commit_silence_ms),
            },
            "guardrails": dict(self.guardrail_metrics),
            "debug": self.debug_snapshot(),
        }

    def is_drained(self) -> bool:
        return bool(self.core.is_drained())

    def recent_pcm_window(
        self,
        *,
        end_ms: int,
        window_ms: int,
        min_t0_ms: int | None = None,
    ) -> bytes:
        return self.core.recent_pcm_window(
            end_ms=end_ms,
            window_ms=window_ms,
            min_t0_ms=min_t0_ms,
        )

    def ingest_audio(self, pcm16le: bytes) -> None:
        self.core.ingest_audio(pcm16le)

    def finalize_input(self) -> None:
        self.core.finalize_input()

    def rollback_inflight_work(self, *, sequence_id: int) -> bool:
        return self.core.rollback_inflight_work(sequence_id=sequence_id)

    def clear_inflight_work(self, *, sequence_id: int) -> bool:
        return self.core.clear_inflight_work(sequence_id=sequence_id)

    def apply_result(self, result: ASRResult) -> ApplyDecision:
        return self.core.apply_result(result)

    def advance_offsets_to(self, *, t1_ms: int, update_last_submitted: bool = False) -> None:
        self.core.advance_offsets_to(t1_ms=t1_ms, update_last_submitted=update_last_submitted)

    def commit_preview_tail(
        self,
        *,
        include_recording_end: bool = True,
        max_t1_ms: int | None = None,
        speech_gate_forced: bool = False,
    ) -> TranscriptSegment | None:
        segment = self.core.commit_preview_tail(
            include_recording_end=include_recording_end,
            max_t1_ms=max_t1_ms,
        )
        if segment is not None and bool(speech_gate_forced):
            self._rt.speech_gate_forced_commit_count = int(max(0, self._rt.speech_gate_forced_commit_count) + 1)
        return segment

    def manual_commit_preview(self) -> PreviewCommitDecision:
        return self.core.manual_commit_preview()

    def handle_speech_activity(
        self,
        *,
        now_mono: float,
        observation: SpeechActivityObservation,
        rearm_from_ms: int,
    ) -> SpeechGateDecision:
        if not bool(self.settings.vad.enabled):
            return SpeechGateDecision(
                allow_work=True,
                reason="vad_disabled",
                previous_state=str(self._rt.speech_gate_state),
                next_state=str(self._rt.speech_gate_state),
            )

        rt = self._rt
        gate = self.settings.speech_gate
        previous_state = str(rt.speech_gate_state)
        cutoff_mono = float(max(0.0, now_mono - (float(gate.rearm_window_ms) / 1000.0)))
        rt.speech_gate_recent_hits_mono = [
            float(ts) for ts in rt.speech_gate_recent_hits_mono if float(ts) >= cutoff_mono
        ]

        if bool(observation.speech_hit):
            rt.last_recent_speech_mono = float(max(0.0, now_mono))
            rt.speech_gate_recent_hits_mono.append(float(now_mono))

        force_commit_requested = False
        silence_elapsed_ms: int | None = None

        if rt.speech_gate_state == "quiet":
            if len(rt.speech_gate_recent_hits_mono) >= int(max(1, gate.rearm_hits)):
                self._set_speech_gate_state(next_state="active", now_mono=now_mono)
                rt.speech_gate_rearm_count = int(max(0, rt.speech_gate_rearm_count) + 1)
            else:
                rt.speech_gate_quiet_skips = int(max(0, rt.speech_gate_quiet_skips) + 1)
                return SpeechGateDecision(
                    allow_work=False,
                    reason="quiet_waiting_rearm",
                    previous_state=previous_state,
                    next_state=str(rt.speech_gate_state),
                )
        elif rt.speech_gate_state == "active":
            silence_elapsed_ms = int(max(0.0, float(now_mono - rt.last_recent_speech_mono) * 1000.0))
            force_threshold_ms = int(max(gate.silence_enter_ms, gate.force_commit_silence_ms))
            if rt.last_recent_speech_mono <= 0.0 or silence_elapsed_ms >= force_threshold_ms:
                force_commit_requested = True
                self._set_speech_gate_state(
                    next_state="quiet",
                    now_mono=now_mono,
                    rearm_from_ms=rearm_from_ms,
                )

        if not bool(observation.speech_hit):
            rt.speech_gate_quiet_skips = int(max(0, rt.speech_gate_quiet_skips) + 1)
            return SpeechGateDecision(
                allow_work=False,
                reason="no_recent_speech",
                force_commit_requested=force_commit_requested,
                previous_state=previous_state,
                next_state=str(rt.speech_gate_state),
                silence_elapsed_ms=silence_elapsed_ms,
            )

        return SpeechGateDecision(
            allow_work=True,
            reason="speech_hit",
            force_commit_requested=force_commit_requested,
            previous_state=previous_state,
            next_state=str(rt.speech_gate_state),
            silence_elapsed_ms=silence_elapsed_ms,
        )

    def observe_speech_activity(
        self,
        *,
        detector: SpeechActivityDetector | None,
        now_mono: float,
        end_ms: int,
        pending_t0_ms: int,
    ) -> SpeechActivityObservation:
        rt = self._rt
        rt.vad_checks = int(max(0, rt.vad_checks) + 1)

        gate_window_ms = int(
            max(
                int(self.settings.speech_gate.rearm_window_ms),
                min(
                    max(int(self.settings.speech_gate.rearm_window_ms), 4000),
                    max(0, int(end_ms) - int(pending_t0_ms)),
                ),
            )
        )
        min_t0_ms = int(max(0, int(pending_t0_ms))) if int(end_ms) > int(pending_t0_ms) else int(max(0, int(end_ms)))
        pcm_recent = self.recent_pcm_window(
            end_ms=int(max(0, end_ms)),
            window_ms=int(max(1, gate_window_ms)),
            min_t0_ms=min_t0_ms,
        )
        if not pcm_recent:
            rt.vad_silence_skips = int(max(0, rt.vad_silence_skips) + 1)
            return SpeechActivityObservation(
                speech_hit=False,
                reason="empty_recent_window",
            )
        if detector is None:
            try:
                self.ensure_vad_ready()
            except Exception as e:
                rt.vad_errors = int(max(0, rt.vad_errors) + 1)
                return SpeechActivityObservation(
                    speech_hit=False,
                    reason="vad_error",
                    error=f"{type(e).__name__}: {e}",
                )
            detector = self._detector
        if detector is None:
            rt.vad_silence_skips = int(max(0, rt.vad_silence_skips) + 1)
            return SpeechActivityObservation(
                speech_hit=False,
                reason="detector_disabled",
            )
        try:
            decision = detector.should_enqueue_pcm16(
                pcm_recent,
                now_mono=now_mono,
                allow_hangover=False,
            )
        except Exception as e:
            rt.vad_errors = int(max(0, rt.vad_errors) + 1)
            return SpeechActivityObservation(
                speech_hit=False,
                reason="vad_error",
                error=f"{type(e).__name__}: {e}",
            )

        allow = bool(decision.get("allow"))
        reason = str(decision.get("reason") or "").strip().lower()
        speech_hit = bool(allow and reason == "speech")
        if speech_hit:
            rt.vad_speech_allows = int(max(0, rt.vad_speech_allows) + 1)
        else:
            rt.vad_silence_skips = int(max(0, rt.vad_silence_skips) + 1)
        return SpeechActivityObservation(
            speech_hit=speech_hit,
            reason=(reason or ("speech" if speech_hit else "silence")),
            speech_ms=int(max(0, int(decision.get("speech_ms") or 0))),
            segments_count=int(max(0, int(decision.get("segments_count") or 0))),
        )

    def maybe_dispatch_work(
        self,
        *,
        now_mono: float,
        force: bool = False,
        detector: SpeechActivityDetector | None = None,
    ) -> DispatchDecision:
        use_force = bool(force)
        speech_observation: SpeechActivityObservation | None = None
        speech_gate_decision: SpeechGateDecision | None = None

        if (not use_force) and bool(self.settings.vad.enabled):
            pending_t0_ms = int(max(self.core.processed_offset_ms, self.core.last_submitted_t1_ms))
            speech_observation = self.observe_speech_activity(
                detector=detector,
                now_mono=now_mono,
                end_ms=int(max(0, self.core.recording_duration_ms)),
                pending_t0_ms=pending_t0_ms,
            )
            if str(speech_observation.error or "").strip():
                return DispatchDecision(
                    reason="vad_error",
                    error=str(speech_observation.error or ""),
                    work_decision=WorkDecision(reason="vad_error"),
                    speech_observation=speech_observation,
                )
            speech_gate_decision = self.handle_speech_activity(
                now_mono=now_mono,
                observation=speech_observation,
                rearm_from_ms=pending_t0_ms,
            )
            if not bool(speech_gate_decision.allow_work):
                return DispatchDecision(
                    reason=str(speech_gate_decision.reason or "speech_gate_blocked"),
                    work_decision=WorkDecision(reason=str(speech_gate_decision.reason or "speech_gate_blocked")),
                    speech_observation=speech_observation,
                    speech_gate_decision=speech_gate_decision,
                )

        work_decision = self.build_work_item(now_mono=now_mono, force=use_force)
        return DispatchDecision(
            reason=str(work_decision.reason or ""),
            work_decision=work_decision,
            speech_observation=speech_observation,
            speech_gate_decision=speech_gate_decision,
        )

    def build_work_item(self, *, now_mono: float, force: bool = False) -> WorkDecision:
        pacing = self.settings.pacing
        use_force = bool(force)
        startup_active = (
            (not use_force)
            and bool(pacing.enabled)
            and int(pacing.policy.startup_duration_ms) > 0
            and int(self.core.recording_duration_ms) < int(pacing.policy.startup_duration_ms)
        )

        if (not use_force) and bool(pacing.enabled):
            if startup_active:
                elapsed_since_emit_ms = int(max(0.0, float(now_mono - float(self._rt.last_emit_mono))) * 1000.0)
                if (self._rt.last_emit_mono > 0.0) and (elapsed_since_emit_ms < int(pacing.policy.startup_emit_ms)):
                    self._rt.emit_interval_skips = int(max(0, self._rt.emit_interval_skips) + 1)
                    return WorkDecision(reason="emit_interval_wait")
            elif not self._consume_pacing_slot(now_mono=now_mono):
                self._rt.pacing_slot_skips = int(max(0, self._rt.pacing_slot_skips) + 1)
                return WorkDecision(reason="pacing_slot_wait")

        decision = self.core.build_work_item(
            force=use_force,
            min_infer_audio_ms_override=(
                int(pacing.policy.startup_min_infer_audio_ms) if startup_active else None
            ),
            min_new_audio_ms_override=(
                int(pacing.policy.startup_min_new_audio_ms) if startup_active else None
            ),
        )
        if decision.work_item is not None:
            self._rt.last_emit_mono = float(max(0.0, now_mono))
        return decision

    def _consume_pacing_slot(self, *, now_mono: float) -> bool:
        safe_now_ms = int(round(float(max(0.0, now_mono)) * 1000.0))
        if self._rt.pacing_epoch_mono_ms <= 0:
            self._rt.pacing_epoch_mono_ms = int(max(0, safe_now_ms))

        interval_ms = int(
            max(
                1,
                max(
                    int(self.settings.pacing.min_emit_interval_ms),
                    int(self.settings.pacing.policy.base_emit_ms),
                ),
            )
        )
        elapsed_ms = int(max(0, safe_now_ms - int(self._rt.pacing_epoch_mono_ms)))
        slot_index = int(elapsed_ms // interval_ms)
        if slot_index <= int(self._rt.pacing_last_slot_index):
            return False
        self._rt.pacing_last_slot_index = int(slot_index)
        return True

    def _set_speech_gate_state(
        self,
        *,
        next_state: Literal["quiet", "active"],
        now_mono: float,
        rearm_from_ms: int | None = None,
    ) -> None:
        rt = self._rt
        if str(next_state) == str(rt.speech_gate_state):
            return
        rt.speech_gate_state = str(next_state)
        rt.speech_gate_state_transitions = int(max(0, rt.speech_gate_state_transitions) + 1)
        if next_state == "quiet":
            rt.speech_gate_recent_hits_mono = []
            rt.last_recent_speech_mono = 0.0
            base_rearm_from_ms = int(max(0, self.core.processed_offset_ms))
            if rearm_from_ms is not None:
                base_rearm_from_ms = int(max(base_rearm_from_ms, int(rearm_from_ms)))
            rt.speech_gate_rearm_from_ms = int(base_rearm_from_ms)
        elif next_state == "active":
            rt.speech_gate_recent_hits_mono = []
            rt.last_recent_speech_mono = float(max(0.0, now_mono))
