from __future__ import annotations

import asyncio
import time
from typing import Protocol
from typing import Sequence

from realtime_asr_engine import ASRResult
from realtime_asr_engine import ASRWorkItem
from realtime_asr_engine import LiveASRRunner
from realtime_asr_engine import TranscriptState


class AudioFrameSource(Protocol):
    async def next_frame(self) -> bytes | None:
        ...


class ASRWorkExecutor(Protocol):
    async def submit(self, work_item: ASRWorkItem) -> None:
        ...

    async def poll_completed(self) -> Sequence[ASRResult]:
        ...


class TranscriptConsumer(Protocol):
    async def publish(self, transcript_state: TranscriptState) -> None:
        ...


async def run_minimal_host_loop(
    *,
    runner: LiveASRRunner,
    audio_source: AudioFrameSource,
    asr_executor: ASRWorkExecutor,
    transcript_consumer: TranscriptConsumer,
    idle_sleep_s: float = 0.01,
) -> TranscriptState:
    input_closed = False

    while True:
        if not input_closed:
            frame = await audio_source.next_frame()
            if frame is None:
                runner.finalize_input()
                input_closed = True
            else:
                runner.ingest_audio(frame)

        dispatch = runner.maybe_dispatch_work(
            now_mono=time.monotonic(),
        )
        if str(dispatch.error or "").strip():
            raise RuntimeError(str(dispatch.error))

        gate_decision = dispatch.speech_gate_decision
        if gate_decision is not None and gate_decision.force_commit_requested:
            runner.commit_preview_tail(
                include_recording_end=False,
                max_t1_ms=(runner.last_submitted_t1_ms if runner.last_submitted_t1_ms > 0 else None),
                speech_gate_forced=True,
            )

        work_item = dispatch.work_decision.work_item
        if work_item is not None:
            await asr_executor.submit(work_item)

        completed = await asr_executor.poll_completed()
        for result in completed:
            runner.apply_result(result)

        await transcript_consumer.publish(runner.transcript_state)

        if input_closed and runner.is_drained():
            return runner.transcript_state

        if work_item is None and not completed and input_closed:
            await asyncio.sleep(float(max(0.0, idle_sleep_s)))
