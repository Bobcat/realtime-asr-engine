# Realtime ASR Engine

`realtime-asr-engine` is a reusable package for incremental ASR over a live audio timeline.

It owns:

- rolling PCM ingest and windowing
- preview and committed transcript state
- pacing and dispatch decisions for new ASR work
- VAD and speech-gate behavior
- input finalization and drain semantics

It does not own:

- transport and websocket protocol
- session lifecycle and persistence
- recording/artifact management
- the concrete ASR backend execution and result-collection strategy

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
