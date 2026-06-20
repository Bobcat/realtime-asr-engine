# Realtime ASR Engine

`realtime-asr-engine` is a reusable package for incremental ASR over a live audio timeline.

It owns:

- rolling PCM ingest and windowing
- preview and committed transcript state
- preview-to-committed transcript heuristics and commit reasons
- pacing and dispatch decisions for new ASR work
- VAD and speech-gate behavior
- input finalization and drain semantics

It does not own:

- transport and websocket protocol
- session lifecycle and persistence
- recording/artifact management
- the concrete ASR backend execution and result-collection strategy

## Index

- [Package Surface](#package-surface)
- [Host Model](#host-model)
- [Commit Heuristics](#commit-heuristics)
- [Host Responsibilities](#host-responsibilities)
- [VAD](#vad)
- [Development](#development)
- [License](#license)

## Package Surface

Main entrypoints:

- `LiveASRRunner`: live engine state machine
- `LiveASRRunnerSettings`: typed runner settings
- `LiveASRRunnerSettings.from_live_config(...)`: package-owned builder from a `live`-style config mapping
- `examples/minimal_host.py`: minimal host loop around the runner

Core modules:

- `src/realtime_asr_engine/core.py`: rolling audio state and ASR result application
- `src/realtime_asr_engine/runner.py`: pacing, VAD, speech gate, dispatch decisions
- `src/realtime_asr_engine/vad_silero.py`: built-in Silero VAD provider

## Host Model

The package assumes a host does roughly this:

1. Create a `LiveASRRunner` with `AudioFormat` and `LiveASRRunnerSettings`.
2. Feed PCM16 audio into the runner.
3. Execute produced `ASRWorkItem`s with the host's own backend integration.
4. Feed completed `ASRResult`s back into the runner.
5. Consume `TranscriptState` and any optional package-exposed runtime payloads it needs.

How a host obtains completed ASR results is intentionally not fixed here. A host can use SSE, callbacks, queues, streams, or any other mechanism.

## Commit Heuristics

The engine keeps committed transcript segments separate from the current preview. Committed segments are treated as stable transcript history. Preview text is provisional and may still be replaced by later ASR results.

ASR results are interpreted by `interpret_asr_result(...)`:

- If an ASR result contains two or more segments, the engine commits all segments except the last one. The last segment remains preview. The commit reason is `rolling_context_commit`.
- If an ASR result contains exactly one segment, that segment normally remains preview. It is committed when `max(segment_duration_ms, infer_window_duration_ms) >= rolling.single_segment_commit_min_ms`. The commit reason is still `rolling_context_commit`, with `single_segment_forced_commit=True`.
- If an ASR result contains no segments but has fallback text, the fallback text becomes preview. No commit is produced.

The repeat guardrail can also force a commit:

- When a non-empty preview reaches the same `preview_audio_end_ms` for `rolling.force_commit_repeats` ASR results, and the current result contains ASR segments, the engine commits all current segments.
- That commit reason is `rolling_context_force_commit_repeats`.
- This path resets preview history after the commit.

Explicit host calls can promote preview to committed text:

- `commit_preview_tail(...)` commits the current preview, or the last remembered preview if the visible preview is empty. The commit reason is `rolling_context_tail_preview_commit`.
- `manual_commit_preview()` commits preview with commit reason `manual_preview_commit`. It also retires overlapping inflight ASR work and returns a restart boundary so the host can continue decoding after the committed preview.

The speech gate does not directly commit text. When VAD is enabled, `maybe_dispatch_work(...)` observes recent speech before building ASR work. If the gate is active and silence lasts at least `max(speech_gate.silence_enter_ms, speech_gate.force_commit_silence_ms)`, it returns a `SpeechGateDecision` with `force_commit_requested=True` and moves the gate back to quiet. The host is responsible for reacting to that flag, typically by calling `commit_preview_tail(...)`.

These actions are not commits:

- `finalize_input()` only allows remaining audio to be dispatched below the usual minimum audio thresholds. It does not commit preview by itself.
- Pacing, VAD, and speech-gate decisions decide whether ASR work should run. They do not produce committed transcript segments on their own.
- The hard-clip guardrail advances offsets and clears preview state when unprocessed audio grows beyond `rolling.max_uncommitted_ms`. It prevents unbounded backlog; it does not create a committed segment.

## Host Responsibilities

A host is still expected to:

- provide audio frames to the runner
- execute produced `ASRWorkItem`s against a concrete backend
- obtain completed `ASRResult`s by a host-chosen mechanism and feed them back into the runner
- consume `TranscriptState` and any optional package-exposed runtime payloads it needs

## VAD

The package includes a Silero-based VAD provider and owns the VAD/speech-gate policy. Hosts do not need to implement a separate VAD adapter.

If VAD is enabled, the built-in provider currently expects `torch`, `numpy`, and the Silero model pathing to be available through a configured Python environment or venv. Those dependencies are loaded lazily when VAD is initialized.

## Development

Install in editable mode:

```bash
pip install -e .
```

Run the package tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
