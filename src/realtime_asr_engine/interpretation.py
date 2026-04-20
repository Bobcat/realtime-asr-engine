from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import TranscriptSegment


def _preview_signature(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass
class PreviewHistoryState:
    last_signature: str = ""
    same_signature_repeats: int = 0
    last_audio_end_ms: int = -1
    same_audio_end_repeats: int = 0
    last_preview_text: str = ""
    last_preview_audio_end_fallback_ms: int = 0
    last_preview_source_t0_ms: int = 0


@dataclass(frozen=True)
class ResultInterpretationSettings:
    single_segment_commit_min_ms: int
    force_commit_repeats: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "single_segment_commit_min_ms",
            int(max(0, int(self.single_segment_commit_min_ms))),
        )
        object.__setattr__(
            self,
            "force_commit_repeats",
            int(max(1, int(self.force_commit_repeats))),
        )


@dataclass(frozen=True)
class InterpretedASRResult:
    committed_segments: tuple[TranscriptSegment, ...]
    preview_text: str
    preview_audio_end_ms: int
    preview_history: PreviewHistoryState
    outcome: Literal["commit", "preview_only", "empty"]
    commit_reason: str = "rolling_context_commit"
    single_segment_forced_commit: bool = False
    force_commit_repeats_applied: bool = False
    reset_preview_history_on_commit: bool = False


def interpret_asr_result(
    *,
    t0_ms: int,
    t1_ms: int,
    segments: list[TranscriptSegment],
    fallback_text: str,
    settings: ResultInterpretationSettings,
    preview_history: PreviewHistoryState | None = None,
) -> InterpretedASRResult:
    history = PreviewHistoryState() if preview_history is None else PreviewHistoryState(**preview_history.__dict__)
    safe_t0_ms = int(max(0, int(t0_ms)))
    safe_t1_ms = int(max(safe_t0_ms, int(t1_ms)))

    committed_segments: tuple[TranscriptSegment, ...] = ()
    preview_text = ""
    preview_audio_end_ms = int(safe_t1_ms)
    single_segment_forced_commit = False
    force_commit_repeats_applied = False
    commit_reason = "rolling_context_commit"

    if len(segments) >= 2:
        committed_segments = tuple(segments[:-1])
        last_segment = segments[-1]
        preview_text = str(last_segment.text or "").strip()
        preview_audio_end_ms = int(max(safe_t0_ms, int(last_segment.t1_ms or safe_t1_ms)))
    elif len(segments) == 1:
        last_segment = segments[0]
        single_text = str(last_segment.text or "").strip()
        single_t0_ms = int(max(safe_t0_ms, int(last_segment.t0_ms or safe_t0_ms)))
        single_t1_ms = int(max(single_t0_ms, int(last_segment.t1_ms or safe_t1_ms)))
        single_duration_ms = int(max(0, single_t1_ms - single_t0_ms))
        infer_window_duration_ms = int(max(0, safe_t1_ms - safe_t0_ms))
        if single_text and max(single_duration_ms, infer_window_duration_ms) >= settings.single_segment_commit_min_ms:
            committed_segments = (last_segment,)
            single_segment_forced_commit = True
            preview_text = ""
            preview_audio_end_ms = int(single_t1_ms)
        else:
            preview_text = single_text
            preview_audio_end_ms = int(single_t1_ms)
    else:
        preview_text = str(fallback_text or "").strip()
        preview_audio_end_ms = int(safe_t1_ms)

    preview_sig = _preview_signature(preview_text)
    if preview_sig:
        if preview_sig == history.last_signature:
            history.same_signature_repeats += 1
        else:
            history.last_signature = preview_sig
            history.same_signature_repeats = 1
        if int(preview_audio_end_ms) == int(history.last_audio_end_ms):
            history.same_audio_end_repeats += 1
        else:
            history.last_audio_end_ms = int(preview_audio_end_ms)
            history.same_audio_end_repeats = 1
    else:
        history.last_signature = ""
        history.same_signature_repeats = 0
        history.last_audio_end_ms = -1
        history.same_audio_end_repeats = 0

    if preview_sig and history.same_audio_end_repeats >= settings.force_commit_repeats and segments:
        last_segment = segments[-1]
        last_text = str(last_segment.text or "").strip()
        if last_text:
            committed_segments = tuple(segments)
            preview_text = ""
            preview_audio_end_ms = int(max(safe_t0_ms, int(last_segment.t1_ms or safe_t1_ms)))
            force_commit_repeats_applied = True
            commit_reason = "rolling_context_force_commit_repeats"

    if preview_text:
        history.last_preview_text = str(preview_text)
        history.last_preview_audio_end_fallback_ms = int(max(0, preview_audio_end_ms))
        history.last_preview_source_t0_ms = int(max(0, safe_t0_ms))

    outcome: Literal["commit", "preview_only", "empty"]
    if committed_segments:
        outcome = "commit"
    elif preview_text:
        outcome = "preview_only"
    else:
        outcome = "empty"

    return InterpretedASRResult(
        committed_segments=committed_segments,
        preview_text=str(preview_text),
        preview_audio_end_ms=int(max(0, preview_audio_end_ms)),
        preview_history=history,
        outcome=outcome,
        commit_reason=str(commit_reason),
        single_segment_forced_commit=bool(single_segment_forced_commit),
        force_commit_repeats_applied=bool(force_commit_repeats_applied),
        reset_preview_history_on_commit=bool(force_commit_repeats_applied),
    )


def reset_preview_history() -> PreviewHistoryState:
    return PreviewHistoryState()
