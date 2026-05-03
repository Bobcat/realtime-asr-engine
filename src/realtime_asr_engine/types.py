from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    text: str
    t0_ms: int
    t1_ms: int
    speaker: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=str(payload.get("segment_id") or ""),
            text=str(payload.get("text") or ""),
            t0_ms=int(max(0, int(payload.get("t0_ms") or 0))),
            t1_ms=int(max(0, int(payload.get("t1_ms") or 0))),
            speaker=str(payload.get("speaker") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": str(self.segment_id),
            "text": str(self.text),
            "t0_ms": int(max(0, int(self.t0_ms))),
            "t1_ms": int(max(0, int(self.t1_ms))),
            "speaker": str(self.speaker),
        }


@dataclass
class PreviewTranscriptState:
    text: str = ""
    audio_end_ms: int = 0


@dataclass
class TranscriptState:
    transcript_revision: int = 0
    committed_segments: list[TranscriptSegment] = field(default_factory=list)
    preview: PreviewTranscriptState = field(default_factory=PreviewTranscriptState)


@dataclass(frozen=True)
class ASRWorkItem:
    sequence_id: int
    t0_ms: int
    t1_ms: int
    pcm16le: bytes = b""
    language: str | None = None


@dataclass(frozen=True)
class ASRResult:
    sequence_id: int
    t0_ms: int
    t1_ms: int
    ok: bool = True
    text: str = ""
    segments: tuple[TranscriptSegment, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class WorkDecision:
    reason: str
    work_item: ASRWorkItem | None = None


@dataclass(frozen=True)
class ApplyDecision:
    reason: str
    applied: bool = False
    committed_segments: tuple[TranscriptSegment, ...] = ()
    preview: PreviewTranscriptState = field(default_factory=PreviewTranscriptState)
    commit_reason: str = ""
    single_segment_forced_commit: bool = False
    force_commit_repeats_applied: bool = False


@dataclass(frozen=True)
class PreviewCommitDecision:
    reason: str
    applied: bool = False
    segment: TranscriptSegment | None = None
    commit_reason: str = ""
    retired_sequence_ids: tuple[int, ...] = ()
    restart_t0_ms: int = 0
