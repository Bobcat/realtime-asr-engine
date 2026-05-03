from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from .audio import AudioFormat
from .interpretation import PreviewHistoryState
from .interpretation import ResultInterpretationSettings
from .interpretation import interpret_asr_result
from .interpretation import reset_preview_history
from .settings import RollingASRSettings
from .types import ASRResult
from .types import ASRWorkItem
from .types import ApplyDecision
from .types import PreviewCommitDecision
from .types import PreviewTranscriptState
from .types import TranscriptSegment
from .types import TranscriptState
from .types import WorkDecision


_MAX_RETIRED_SEQUENCE_IDS = 32


@dataclass
class _RuntimeState:
    rolling_pcm: bytearray = field(default_factory=bytearray)
    pcm_base_ms: int = 0
    processed_offset_ms: int = 0
    decode_offset_ms: int = 0
    recording_duration_ms: int = 0
    infer_seq_next: int = 0
    last_submitted_t1_ms: int = 0
    inflight_previous_last_submitted_t1_ms: int = 0
    inflight_work: ASRWorkItem | None = None
    retired_sequence_ids: set[int] = field(default_factory=set)
    input_finalized: bool = False
    hard_clip_count: int = 0
    hard_clip_dropped_audio_ms: int = 0
    buffer_trim_count: int = 0
    buffer_trim_dropped_audio_ms: int = 0
    work_decision_counts: dict[str, int] = field(default_factory=dict)
    apply_decision_counts: dict[str, int] = field(default_factory=dict)
    commit_reason_counts: dict[str, int] = field(default_factory=dict)


def _increment_reason_count(counts: dict[str, int], reason: str) -> None:
    key = str(reason or "").strip() or "unknown"
    counts[key] = int(max(0, int(counts.get(key) or 0)) + 1)


class RollingASRCore:
    def __init__(
        self,
        *,
        audio_format: AudioFormat,
        settings: RollingASRSettings,
        language: str | None = None,
    ) -> None:
        self.audio_format = audio_format
        self.settings = settings.normalized()
        self.language = language
        self.transcript_state = TranscriptState()
        self.preview_history = PreviewHistoryState()
        self._rt = _RuntimeState()

    @property
    def input_finalized(self) -> bool:
        return bool(self._rt.input_finalized)

    @property
    def has_inflight_work(self) -> bool:
        return self._rt.inflight_work is not None

    @property
    def recording_duration_ms(self) -> int:
        return int(self._rt.recording_duration_ms)

    @property
    def pcm_base_ms(self) -> int:
        return int(self._rt.pcm_base_ms)

    @property
    def processed_offset_ms(self) -> int:
        return int(self._rt.processed_offset_ms)

    @property
    def decode_offset_ms(self) -> int:
        return int(self._rt.decode_offset_ms)

    @property
    def last_submitted_t1_ms(self) -> int:
        return int(self._rt.last_submitted_t1_ms)

    @property
    def guardrail_metrics(self) -> dict[str, int]:
        return {
            "hard_clip_count": int(self._rt.hard_clip_count),
            "hard_clip_dropped_audio_ms": int(self._rt.hard_clip_dropped_audio_ms),
            "buffer_trim_count": int(self._rt.buffer_trim_count),
            "buffer_trim_dropped_audio_ms": int(self._rt.buffer_trim_dropped_audio_ms),
        }

    def debug_snapshot(self) -> dict[str, Any]:
        rt = self._rt
        inflight_payload: dict[str, Any] | None = None
        if rt.inflight_work is not None:
            inflight_payload = {
                "sequence_id": int(rt.inflight_work.sequence_id),
                "t0_ms": int(rt.inflight_work.t0_ms),
                "t1_ms": int(rt.inflight_work.t1_ms),
                "audio_bytes": int(len(rt.inflight_work.pcm16le or b"")),
                "language": str(rt.inflight_work.language or ""),
            }

        preview_text = str(self.transcript_state.preview.text or "").strip()
        if not preview_text:
            preview_text = str(self.preview_history.last_preview_text or "").strip()
        preview_audio_end_ms = int(max(0, int(self.transcript_state.preview.audio_end_ms or 0)))
        if preview_audio_end_ms <= 0:
            preview_audio_end_ms = int(max(0, int(self.preview_history.last_preview_audio_end_fallback_ms or 0)))

        return {
            "state": {
                "recording_duration_ms": int(rt.recording_duration_ms),
                "pcm_base_ms": int(rt.pcm_base_ms),
                "processed_offset_ms": int(rt.processed_offset_ms),
                "decode_offset_ms": int(rt.decode_offset_ms),
                "last_submitted_t1_ms": int(rt.last_submitted_t1_ms),
                "buffer_audio_ms": int(max(0, int(rt.recording_duration_ms) - int(rt.pcm_base_ms))),
                "unprocessed_audio_ms": int(max(0, int(rt.recording_duration_ms) - int(rt.processed_offset_ms))),
                "input_finalized": bool(rt.input_finalized),
                "inflight": inflight_payload,
                "preview_chars": int(len(preview_text)),
                "preview_audio_end_ms": int(preview_audio_end_ms),
            },
            "reason_counts": {
                "work_decision": dict(rt.work_decision_counts),
                "apply_decision": dict(rt.apply_decision_counts),
                "commit_reason": dict(rt.commit_reason_counts),
            },
        }

    def is_drained(self) -> bool:
        return (
            bool(self._rt.input_finalized)
            and self._rt.inflight_work is None
            and int(self._rt.processed_offset_ms) >= int(self._rt.recording_duration_ms)
        )

    def recent_pcm_window(
        self,
        *,
        end_ms: int,
        window_ms: int,
        min_t0_ms: int | None = None,
    ) -> bytes:
        rt = self._rt
        safe_end_ms = int(max(0, end_ms))
        safe_window_ms = int(max(1, window_ms))
        t0_ms = int(max(rt.pcm_base_ms, safe_end_ms - safe_window_ms))
        if min_t0_ms is not None:
            t0_ms = int(max(t0_ms, int(max(0, min_t0_ms))))
        t1_ms = int(max(t0_ms, safe_end_ms))
        start_b = self.audio_format.ms_to_byte_offset(int(max(0, t0_ms - rt.pcm_base_ms)))
        end_b = self.audio_format.ms_to_byte_offset(int(max(0, t1_ms - rt.pcm_base_ms)))
        end_b = int(max(start_b, min(end_b, len(rt.rolling_pcm))))
        if end_b <= start_b:
            return b""
        return bytes(rt.rolling_pcm[start_b:end_b])

    def ingest_audio(self, pcm16le: bytes) -> None:
        if self._rt.input_finalized:
            raise RuntimeError("audio_input_already_finalized")
        raw = bytes(pcm16le or b"")
        if not raw:
            return
        align = int(max(1, self.audio_format.sample_width_bytes))
        if (len(raw) % align) != 0:
            raw = raw[: len(raw) - (len(raw) % align)]
        if not raw:
            return
        self._rt.rolling_pcm.extend(raw)
        self._rt.recording_duration_ms = int(
            self._rt.pcm_base_ms + self.audio_format.bytes_to_ms(len(self._rt.rolling_pcm))
        )

    def finalize_input(self) -> None:
        self._rt.input_finalized = True

    def rollback_inflight_work(self, *, sequence_id: int) -> bool:
        rt = self._rt
        inflight = rt.inflight_work
        if inflight is None or int(inflight.sequence_id) != int(sequence_id):
            return False
        rt.last_submitted_t1_ms = int(max(0, rt.inflight_previous_last_submitted_t1_ms))
        rt.inflight_previous_last_submitted_t1_ms = 0
        rt.inflight_work = None
        return True

    def clear_inflight_work(self, *, sequence_id: int) -> bool:
        rt = self._rt
        inflight = rt.inflight_work
        if inflight is None or int(inflight.sequence_id) != int(sequence_id):
            return False
        rt.inflight_previous_last_submitted_t1_ms = 0
        rt.inflight_work = None
        return True

    def advance_offsets_to(self, *, t1_ms: int, update_last_submitted: bool = False) -> None:
        rt = self._rt
        safe_t1_ms = int(max(0, int(t1_ms)))
        rt.processed_offset_ms = int(max(rt.processed_offset_ms, safe_t1_ms))
        rt.decode_offset_ms = int(max(rt.decode_offset_ms, safe_t1_ms))
        if update_last_submitted:
            rt.last_submitted_t1_ms = int(max(rt.last_submitted_t1_ms, safe_t1_ms))
        self._maybe_trim_pcm_buffer()

    def build_work_item(
        self,
        *,
        force: bool = False,
        min_infer_audio_ms_override: int | None = None,
        min_new_audio_ms_override: int | None = None,
    ) -> WorkDecision:
        rt = self._rt
        if rt.inflight_work is not None:
            return self._record_work_decision(WorkDecision(reason="already_inflight"))

        end_ms = int(max(0, rt.recording_duration_ms))
        self._maybe_apply_hard_clip(end_ms=end_ms)
        if end_ms <= rt.processed_offset_ms:
            if rt.input_finalized:
                return self._record_work_decision(WorkDecision(reason="input_drained"))
            return self._record_work_decision(WorkDecision(reason="no_unprocessed_audio"))

        use_force = bool(force or rt.input_finalized)
        unprocessed_ms = int(max(0, end_ms - rt.processed_offset_ms))
        effective_min_infer_audio_ms = int(
            self.settings.min_infer_audio_ms
            if min_infer_audio_ms_override is None
            else max(0, int(min_infer_audio_ms_override))
        )
        if (not use_force) and (unprocessed_ms < effective_min_infer_audio_ms):
            return self._record_work_decision(WorkDecision(reason="insufficient_unprocessed_audio"))

        effective_min_new_audio_ms = int(
            self.settings.min_new_audio_ms
            if min_new_audio_ms_override is None
            else max(0, int(min_new_audio_ms_override))
        )
        if (not use_force) and rt.last_submitted_t1_ms > 0:
            delta_new_audio_ms = int(max(0, end_ms - rt.last_submitted_t1_ms))
            if delta_new_audio_ms < effective_min_new_audio_ms:
                return self._record_work_decision(WorkDecision(reason="insufficient_new_audio"))

        infer_t0_ms = int(max(rt.processed_offset_ms, rt.decode_offset_ms, rt.pcm_base_ms))
        infer_t1_ms = int(max(infer_t0_ms, end_ms))
        infer_window_ms = int(max(0, infer_t1_ms - infer_t0_ms))
        if infer_window_ms > self.settings.max_decode_window_ms:
            infer_t1_ms = int(max(infer_t0_ms, infer_t0_ms + self.settings.max_decode_window_ms))

        if (not use_force) and infer_t1_ms <= rt.last_submitted_t1_ms:
            if infer_window_ms >= self.settings.max_decode_window_ms:
                slide_ms = int(max(1, effective_min_new_audio_ms))
                rt.decode_offset_ms = int(
                    max(
                        rt.processed_offset_ms,
                        infer_t0_ms + slide_ms,
                        int(rt.last_submitted_t1_ms) - int(self.settings.max_decode_window_ms) + slide_ms,
                    )
                )
            return self._record_work_decision(WorkDecision(reason="window_no_progress"))

        start_b = self.audio_format.ms_to_byte_offset(int(max(0, infer_t0_ms - rt.pcm_base_ms)))
        end_b = self.audio_format.ms_to_byte_offset(int(max(0, infer_t1_ms - rt.pcm_base_ms)))
        end_b = int(min(end_b, len(rt.rolling_pcm)))
        if end_b <= start_b:
            return self._record_work_decision(WorkDecision(reason="empty_audio_window"))
        pcm = bytes(rt.rolling_pcm[start_b:end_b])
        if not pcm:
            return self._record_work_decision(WorkDecision(reason="empty_audio_window"))

        work_item = ASRWorkItem(
            sequence_id=int(max(0, rt.infer_seq_next)),
            t0_ms=int(infer_t0_ms),
            t1_ms=int(infer_t1_ms),
            pcm16le=pcm,
            language=self.language,
        )
        rt.infer_seq_next = int(work_item.sequence_id + 1)
        rt.inflight_previous_last_submitted_t1_ms = int(max(0, rt.last_submitted_t1_ms))
        rt.last_submitted_t1_ms = int(max(rt.last_submitted_t1_ms, infer_t1_ms))
        rt.inflight_work = work_item
        return self._record_work_decision(WorkDecision(reason="work_item_ready", work_item=work_item))

    def apply_result(self, result: ASRResult) -> ApplyDecision:
        rt = self._rt
        sequence_id = int(result.sequence_id)
        if sequence_id in rt.retired_sequence_ids:
            rt.retired_sequence_ids.discard(sequence_id)
            return self._record_apply_decision(ApplyDecision(reason="retired_result", applied=False))
        inflight = rt.inflight_work
        if inflight is None:
            return self._record_apply_decision(ApplyDecision(reason="no_inflight_work", applied=False))
        if sequence_id != int(inflight.sequence_id):
            if sequence_id < int(inflight.sequence_id):
                return self._record_apply_decision(ApplyDecision(reason="stale_result", applied=False))
            return self._record_apply_decision(ApplyDecision(reason="unexpected_result", applied=False))

        rt.inflight_previous_last_submitted_t1_ms = 0
        rt.inflight_work = None

        if not bool(result.ok):
            rt.processed_offset_ms = int(max(rt.processed_offset_ms, int(result.t1_ms)))
            rt.decode_offset_ms = int(max(rt.decode_offset_ms, int(result.t1_ms)))
            self._maybe_trim_pcm_buffer()
            return self._record_apply_decision(ApplyDecision(reason="error_result", applied=False))

        interpreted = interpret_asr_result(
            t0_ms=int(result.t0_ms),
            t1_ms=int(result.t1_ms),
            segments=list(result.segments),
            fallback_text=str(result.text or ""),
            settings=ResultInterpretationSettings(
                single_segment_commit_min_ms=self.settings.single_segment_commit_min_ms,
                force_commit_repeats=self.settings.force_commit_repeats,
            ),
            preview_history=self.preview_history,
        )
        self.preview_history = interpreted.preview_history

        if interpreted.committed_segments:
            committed_segments = self._normalize_committed_segments(
                interpreted.committed_segments,
                commit_t0_ms=int(max(0, rt.processed_offset_ms, int(result.t0_ms))),
                commit_t1_ms=int(max(0, int(result.t1_ms))),
                extend_last_to_window=bool(
                    interpreted.single_segment_forced_commit or interpreted.force_commit_repeats_applied
                ),
            )
            if committed_segments:
                self.transcript_state.committed_segments.extend(committed_segments)
                self.transcript_state.transcript_revision += 1
                commit_t1_ms = int(max(seg.t1_ms for seg in committed_segments))
                rt.processed_offset_ms = int(max(rt.processed_offset_ms, commit_t1_ms))
                rt.decode_offset_ms = int(max(rt.decode_offset_ms, commit_t1_ms))
                if interpreted.preview_text:
                    self.transcript_state.preview = PreviewTranscriptState(
                        text=str(interpreted.preview_text),
                        audio_end_ms=int(max(0, interpreted.preview_audio_end_ms)),
                    )
                else:
                    self.transcript_state.preview = PreviewTranscriptState()
                    self.preview_history = reset_preview_history()
                if interpreted.reset_preview_history_on_commit:
                    self.preview_history = reset_preview_history()
                self._maybe_trim_pcm_buffer()
                return self._record_apply_decision(
                    ApplyDecision(
                        reason="commit_applied",
                        applied=True,
                        committed_segments=tuple(committed_segments),
                        preview=self.transcript_state.preview,
                        commit_reason=str(interpreted.commit_reason),
                        single_segment_forced_commit=bool(interpreted.single_segment_forced_commit),
                        force_commit_repeats_applied=bool(interpreted.force_commit_repeats_applied),
                    )
                )

        if interpreted.preview_text:
            self.transcript_state.preview = PreviewTranscriptState(
                text=str(interpreted.preview_text),
                audio_end_ms=int(max(0, interpreted.preview_audio_end_ms)),
            )
            return self._record_apply_decision(
                ApplyDecision(
                    reason="preview_applied",
                    applied=True,
                    preview=self.transcript_state.preview,
                )
            )

        return self._record_apply_decision(
            ApplyDecision(
                reason="empty_result",
                applied=False,
                preview=self.transcript_state.preview,
            )
        )

    def commit_preview_tail(
        self,
        *,
        include_recording_end: bool = True,
        max_t1_ms: int | None = None,
    ) -> TranscriptSegment | None:
        return self._commit_preview(
            include_recording_end=include_recording_end,
            max_t1_ms=max_t1_ms,
            commit_reason="rolling_context_tail_preview_commit",
        )

    def manual_commit_preview(self) -> PreviewCommitDecision:
        commit_reason = "manual_preview_commit"
        segment = self._commit_preview(
            include_recording_end=False,
            max_t1_ms=None,
            commit_reason=commit_reason,
        )
        if segment is None:
            return PreviewCommitDecision(
                reason="no_preview",
                commit_reason=commit_reason,
                restart_t0_ms=int(max(0, self._rt.last_submitted_t1_ms)),
            )

        rt = self._rt
        commit_t1_ms = int(max(0, int(segment.t1_ms)))
        inflight = rt.inflight_work
        retired_sequence_ids: list[int] = []
        if inflight is not None and int(inflight.t0_ms) < commit_t1_ms:
            retired_sequence_ids.append(int(inflight.sequence_id))
            self._retire_sequence_id(int(inflight.sequence_id))
            rt.inflight_previous_last_submitted_t1_ms = 0
            rt.inflight_work = None
        if inflight is None or retired_sequence_ids:
            rt.last_submitted_t1_ms = int(commit_t1_ms)
        return PreviewCommitDecision(
            reason="manual_preview_committed",
            applied=True,
            segment=segment,
            commit_reason=commit_reason,
            retired_sequence_ids=tuple(retired_sequence_ids),
            restart_t0_ms=int(commit_t1_ms),
        )

    def _commit_preview(
        self,
        *,
        include_recording_end: bool,
        max_t1_ms: int | None,
        commit_reason: str,
    ) -> TranscriptSegment | None:
        rt = self._rt
        preview = self.transcript_state.preview
        preview_text = str(preview.text or "").strip()
        if not preview_text:
            preview_text = str(self.preview_history.last_preview_text or "").strip()
        if not preview_text:
            return None

        preview_audio_end_ms = int(max(0, int(preview.audio_end_ms or 0)))
        if preview_audio_end_ms <= 0:
            preview_audio_end_ms = int(max(0, self.preview_history.last_preview_audio_end_fallback_ms))

        commit_t0_ms = int(max(0, rt.processed_offset_ms, self.preview_history.last_preview_source_t0_ms))
        commit_t1_candidates = [int(commit_t0_ms), int(preview_audio_end_ms)]
        if include_recording_end:
            commit_t1_candidates.append(int(max(0, rt.recording_duration_ms)))
        commit_t1_ms = int(max(commit_t1_candidates))
        if max_t1_ms is not None:
            commit_t1_ms = int(min(commit_t1_ms, int(max(0, max_t1_ms))))
        if commit_t1_ms <= commit_t0_ms:
            if include_recording_end:
                commit_t1_ms = int(max(commit_t0_ms + 1, rt.recording_duration_ms))
            else:
                return None

        segment_index = int(max(0, len(self.transcript_state.committed_segments))) + 1
        segment = TranscriptSegment(
            segment_id=f"s{segment_index:04d}",
            text=str(preview_text),
            t0_ms=int(commit_t0_ms),
            t1_ms=int(commit_t1_ms),
        )
        self.transcript_state.committed_segments.append(segment)
        self.transcript_state.transcript_revision += 1
        self.transcript_state.preview = PreviewTranscriptState()
        rt.processed_offset_ms = int(max(rt.processed_offset_ms, commit_t1_ms))
        rt.decode_offset_ms = int(max(rt.decode_offset_ms, commit_t1_ms))
        self.preview_history = reset_preview_history()
        self._maybe_trim_pcm_buffer()
        _increment_reason_count(rt.commit_reason_counts, commit_reason)
        return segment

    def _record_work_decision(self, decision: WorkDecision) -> WorkDecision:
        _increment_reason_count(self._rt.work_decision_counts, str(decision.reason or ""))
        return decision

    def _record_apply_decision(self, decision: ApplyDecision) -> ApplyDecision:
        _increment_reason_count(self._rt.apply_decision_counts, str(decision.reason or ""))
        if str(decision.commit_reason or "").strip():
            _increment_reason_count(self._rt.commit_reason_counts, str(decision.commit_reason or ""))
        return decision

    def _retire_sequence_id(self, sequence_id: int) -> None:
        rt = self._rt
        rt.retired_sequence_ids.add(int(sequence_id))
        while len(rt.retired_sequence_ids) > _MAX_RETIRED_SEQUENCE_IDS:
            rt.retired_sequence_ids.discard(min(rt.retired_sequence_ids))

    def _normalize_committed_segments(
        self,
        segments: tuple[TranscriptSegment, ...],
        *,
        commit_t0_ms: int,
        commit_t1_ms: int,
        extend_last_to_window: bool,
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        for segment in segments:
            segment_text = str(segment.text or "").strip()
            if not segment_text:
                continue
            seg_t0_ms = int(max(commit_t0_ms, int(segment.t0_ms)))
            seg_t1_ms = int(max(seg_t0_ms, int(segment.t1_ms)))
            if seg_t1_ms <= commit_t0_ms:
                continue
            normalized.append(
                TranscriptSegment(
                    segment_id=str(segment.segment_id or ""),
                    text=segment_text,
                    t0_ms=seg_t0_ms,
                    t1_ms=seg_t1_ms,
                    speaker=str(segment.speaker or ""),
                )
            )
        if not normalized:
            return []

        first = normalized[0]
        last = normalized[-1]
        normalized[0] = TranscriptSegment(
            segment_id=first.segment_id,
            text=first.text,
            t0_ms=int(commit_t0_ms),
            t1_ms=first.t1_ms,
            speaker=first.speaker,
        )
        effective_commit_t1_ms = int(max(commit_t0_ms, commit_t1_ms))
        if not extend_last_to_window:
            effective_commit_t1_ms = int(max(commit_t0_ms, last.t1_ms))
        else:
            effective_commit_t1_ms = int(max(effective_commit_t1_ms, last.t1_ms))
        normalized[-1] = TranscriptSegment(
            segment_id=last.segment_id,
            text=last.text,
            t0_ms=last.t0_ms,
            t1_ms=int(max(last.t1_ms, effective_commit_t1_ms)),
            speaker=last.speaker,
        )
        return normalized

    def _maybe_trim_pcm_buffer(self) -> None:
        rt = self._rt
        committed_in_buffer_ms = int(max(0, int(rt.processed_offset_ms) - int(rt.pcm_base_ms)))
        if committed_in_buffer_ms < self.settings.buffer_trim_threshold_ms:
            return
        target_base_ms = int(
            min(
                rt.processed_offset_ms,
                int(rt.pcm_base_ms) + int(self.settings.buffer_trim_drop_ms),
            )
        )
        dropped_ms = self._drop_pcm_prefix_to_ms(target_base_ms=target_base_ms)
        if dropped_ms > 0:
            rt.buffer_trim_count = int(max(0, int(rt.buffer_trim_count)) + 1)
            rt.buffer_trim_dropped_audio_ms = int(
                max(0, int(rt.buffer_trim_dropped_audio_ms)) + int(max(0, dropped_ms))
            )

    def _maybe_apply_hard_clip(self, *, end_ms: int) -> None:
        rt = self._rt
        unprocessed_ms = int(max(0, int(end_ms) - int(rt.processed_offset_ms)))
        if unprocessed_ms <= self.settings.max_uncommitted_ms:
            return
        clip_target_ms = int(
            max(
                rt.processed_offset_ms,
                int(end_ms) - int(self.settings.hard_clip_keep_tail_ms),
            )
        )
        if clip_target_ms <= rt.processed_offset_ms:
            return
        dropped_uncommitted_ms = int(max(0, clip_target_ms - rt.processed_offset_ms))
        rt.processed_offset_ms = int(clip_target_ms)
        rt.decode_offset_ms = int(max(rt.decode_offset_ms, clip_target_ms))
        dropped_buffer_ms = self._drop_pcm_prefix_to_ms(target_base_ms=clip_target_ms)
        rt.hard_clip_count = int(max(0, int(rt.hard_clip_count)) + 1)
        rt.hard_clip_dropped_audio_ms = int(
            max(0, int(rt.hard_clip_dropped_audio_ms))
            + int(max(dropped_uncommitted_ms, dropped_buffer_ms))
        )
        self.transcript_state.preview = PreviewTranscriptState()
        self.preview_history = reset_preview_history()

    def _drop_pcm_prefix_to_ms(self, *, target_base_ms: int) -> int:
        rt = self._rt
        safe_target_ms = int(max(rt.pcm_base_ms, int(target_base_ms)))
        if safe_target_ms <= rt.pcm_base_ms:
            return 0
        drop_window_ms = int(max(0, safe_target_ms - rt.pcm_base_ms))
        drop_bytes = int(max(0, min(self.audio_format.ms_to_byte_offset(drop_window_ms), len(rt.rolling_pcm))))
        if drop_bytes <= 0:
            return 0
        del rt.rolling_pcm[:drop_bytes]
        dropped_ms = int(self.audio_format.bytes_to_ms(drop_bytes))
        if dropped_ms <= 0:
            dropped_ms = int(max(1, drop_window_ms))
        rt.pcm_base_ms = int(rt.pcm_base_ms + dropped_ms)
        if rt.processed_offset_ms < rt.pcm_base_ms:
            rt.processed_offset_ms = int(rt.pcm_base_ms)
        if rt.decode_offset_ms < rt.pcm_base_ms:
            rt.decode_offset_ms = int(rt.pcm_base_ms)
        rt.recording_duration_ms = int(rt.pcm_base_ms + self.audio_format.bytes_to_ms(len(rt.rolling_pcm)))
        return int(max(0, dropped_ms))
