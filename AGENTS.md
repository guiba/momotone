# AGENTS.md - AI Assistant Context for gesture_rust

This document provides structural and architectural context to guide AI assistants in working with this codebase.

## Project Overview
`gesture_rust` is a real-time hand gesture recognition engine in Rust: OpenCV captures webcam frames, an **ONNX two-stage hand pipeline** (BlazeHand detector → landmark detector, via `ort`) produces a hand box plus 21 landmarks, a **strategy** turns those landmarks into a `GestureEvent`, and the **event bus** dispatches it to sinks (audio, MIDI) while the visualizer renders an HUD.

> **Current status (Oct 2026):** the vision pipeline works end-to-end and `cargo check` passes (14 dead-code warnings, no errors). The runtime is a **single synchronous loop in `main.rs`** — `tokio` is still in `Cargo.toml` but is not used anywhere in `src/`. The detector stage is solid; the landmarker stage runs and returns 21 landmarks but is only lightly validated. Several modules (`sources/mock.rs`, unused strategies, the `Source`/`Tracker` traits) are scaffolding from earlier iterations.

## Tech Stack
- **Language**: Rust (Edition 2021)
- **Vision / inference**: `opencv` 0.100 (`clang-runtime` feature), `ort` 2.0.0-rc.13 (ONNX Runtime), `ndarray` 0.17
- **Audio**: `cpal` 0.17 (`Synthesizer` with a `Voice` sine oscillator)
- **MIDI**: `midir` 0.11 (`MidiOutputEngine`)
- **Messaging**: `crossbeam-channel` (unbounded MPMC) — the event bus
- **CLI**: `clap` 4.5 (derive); `serde`/`serde_json` on the event types
- **Async**: `tokio` declared but **unused**; concurrency is `std` + `Mutex`/`Arc`
- **Error handling**: `anyhow`

## Build Environment
- `.cargo/config.toml` forces `LIBCLANG_PATH` / `DYLD_LIBRARY_PATH` to `/opt/homebrew/opt/llvm/lib` (Homebrew LLVM) so the `opencv` build script finds clang. Do not remove it on macOS.
- OpenCV is built by the crate's build script on first compile; `target/` is ~1.7 GB and is gitignored.
- Quick verification loop: `cargo check` (≈4 s warm) rather than a full build.

## Core Architecture & Data Flow
Everything is orchestrated synchronously inside the `loop` in `src/main.rs`:

```
WebcamSource.get_frame()  ->  HandTrackerImpl.process_frame()  ->  Strategy.process()
        (opencv)                 (ort: detector + landmarker)          (GestureEvent)
                                              |
                                        EventBus.dispatch()  ->  Sink.handle_event()
                                              |                        (Synthesizer / MidiOutputEngine)
                                        Visualizer.draw_skeleton() + draw_hud() + imshow
```

1.  **Input Phase**:
    *   `src/vision/camera.rs`: `WebcamSource::get_frame()` returns an OpenCV `Mat` (BGR). It does **not** implement the `Source` trait — that trait is currently unused scaffolding.
    *   `src/sources/mock.rs`: `MockSource` / `MockSourceImpl` are **dead code**. Mock mode is generated inline in `main.rs` (`Mode::Mock` branch, one event every 2 s).

2.  **Vision Phase** (`src/vision/tracker.rs`, `HandTrackerImpl`):
    *   Stage 1 — `run_detector()`: largest centred square of the frame resampled to **192×192** via `sample_affine_to_tensor()` (hand-written bilinear sampling, BGR→RGB, `/255`). Outputs 2016 boxes (18 values per anchor) + 2016 raw scores; scores are **sigmoid**-ed, argmax wins, gated by `SCORE_THRESHOLD = 0.3`.
    *   Anchor offsets are converted to frame pixels and the palm box is **expanded ×2** to cover the whole hand (`r#box` = `(x, y, w, h)` in frame pixels).
    *   Stage 2 — `run_landmarks()`: crops the box (side = max(w,h) × 1.2) into **224×224** and runs the landmark model; the 63 floats are mapped back to **frame pixel coordinates** (`Landmark { x, y, z }`). `--debug` prints the winning anchor, the landmark bbox and all 21 landmark coordinates.
    *   The tracker computes geometry only — it draws nothing. All overlays live in `src/visualizer/mod.rs`.
    *   Results are `HandDetectionResult { hand_landmarks: Option<Vec<Landmark>>, r#box: Option<(f32,f32,f32,f32)>, score: f32 }` (defined in `src/core/mod.rs`).

3.  **Strategy Phase** (`src/strategies/mod.rs`): implements `Strategy` from `src/core/mod.rs`.
    *   `FingerPositionStrategy` — the one used by `main.rs`. Index-tip Y normalised against the hand box → note **36–84**; thumb↔index distance → velocity/volume 0–1, where fingers spread apart is loud and a pinch is silent (saturates at `MAX_SPREAD = 0.6` of the hand box height). Unit tests in `src/strategies/mod.rs` cover the direction of the mapping.
    *   `SimplePinchStrategy` (thumb/index distance < 0.05) and `FrequencyStrategy` exist but are **not wired in**.

4.  **Orchestration**:
    *   `src/core/event_bus.rs`: `EventBus::new()` returns `(EventBus, Receiver<GestureEvent>)`; `dispatch()` sends on a crossbeam channel. `main.rs` drains with `try_recv()` each iteration.
    *   `src/core/types.rs`: `EventType { Continuous, Discrete }`, `GesturePayload { Signal { frequency, amplitude } | Note { note, velocity } }`, `GestureEvent { event_type, payload, text, color }` plus constructors `new_signal`, `new_note`, `no_gesture` (unused).

5.  **Output Phase** (implement `Sink`: `start`/`stop`/`handle_event`):
    *   `src/audio/synthesizer.rs`: `Synthesizer` — cpal output stream; shared `Arc<Mutex<SynthesizerState>>` between the callback and the event handler. `cpal::Stream` is wrapped in `StreamWrapper` with `unsafe impl Send/Sync`.
    *   `src/midi/mod.rs`: `MidiOutputEngine` — connects to a port whose name contains `--port` (default `OP-1`). Note payload is sent as `0x90, note + 60, velocity*127`.
    *   `src/visualizer/mod.rs`: all frame overlays — `draw_skeleton()` (bones + joint dots) and `draw_hud()` (gesture text / status). Only landmark indices from MediaPipe's `HandLandmarksConnections` are drawn; the legacy `solutions.hands` cross-links ((9,4), (4,7), (4,10), (9,2), (13,5), (14,5)) are intentionally omitted because they assume an upright palm and render as stray lines on rotated hands.

## Key Module Reference
| Module | Responsibility |
| :--- | :--- |
| `src/main.rs` | Entry point, `Args` (clap), sink setup, and the camera → tracker → strategy → sinks → HUD loop. |
| `src/core/mod.rs` | Traits (`Sink`, `Strategy`, and the unused `Source`/`Tracker`) + `Landmark`, `HandDetectionResult`. |
| `src/core/types.rs` | `GestureEvent`, `GesturePayload`, `EventType`. |
| `src/core/event_bus.rs` | crossbeam-channel event hub. |
| `src/vision/camera.rs` | OpenCV webcam capture (`WebcamSource`). |
| `src/vision/tracker.rs` | ONNX hand detection + landmark extraction (no drawing). |
| `src/strategies/mod.rs` | Landmark → `GestureEvent` mappings. |
| `src/audio/synthesizer.rs` | cpal sine-wave audio sink. |
| `src/midi/mod.rs` | midir MIDI sink. |
| `src/visualizer/mod.rs` | Hand skeleton (`HAND_CONNECTIONS`, the 21-edge MediaPipe `HandLandmarksConnections` topology) and HUD overlay. |
| `src/sources/mock.rs` | Unused mock source (legacy). |

## Running It
Models live in `gesture_rust/assets/` and are **gitignored** — they must be present locally or the tracker fails at startup. They come from the Unity/Sentis BlazeHand package (`mediapipe_hand_gesture-onnx-float.zip`; see `assets/new_readme.md`). Defaults:
`--detector-path assets/hand_detector.onnx`, `--landmarker-path assets/hand_landmarks_detector.onnx`, `--anchors-path assets/anchors.csv` (2016 anchors, normalised 0–1).

```bash
cargo run                                  # signal mode, audio output
cargo run -- --mode midi --port OP-1       # MIDI output
cargo run -- --mode mock                   # no hand needed, simulated events
cargo run -- --debug                       # tracker diagnostics to stderr

# Headless detection check without a webcam (best way to verify the box):
cargo run -- --image <any-photo-with-a-hand.png> --dump-frame output_frame.png --max-frames 1
```

Useful `Args`: `--mode signal|midi|mock`, `--amplitude`, `--port`, `--image`, `--dump-frame`, `--max-frames`, `--debug`.

## Known Gaps / Things To Fix
- **MIDI range bug**: `FingerPositionStrategy` emits notes 36–84 and `MidiOutputEngine` adds +60 → values up to 144, outside the 0–127 MIDI range. Either drop the +60 offset or narrow the strategy range.
- No notes are ever sent **off**; note-on events repeat with no release.
- `Landmark.z` is read nowhere; `SimplePinchStrategy`/`FrequencyStrategy` are never constructed; `no_gesture()` is unused; the hand box is computed but never drawn.
- No tests, no lint/CI; `cargo check` warnings are the only signal.
- The loop is single-threaded: detection runs on the same thread as rendering/audio dispatch.

## Development Guidelines for AI Assistants

### 1. Design Patterns
- **Trait-Based Abstraction**: To add a new output (e.g. OSC), implement `Sink` (`src/core/mod.rs`) and push it into the `sinks` vec in `main.rs`. New gesture logic means a new `Strategy` impl in `src/strategies/mod.rs`.
- **Coordinate discipline**: landmarks and boxes are always in **frame pixel space**, not normalised. Convert explicitly when feeding a model (192×192 detector, 224×224 landmarker) and convert back when drawing.
- **Error Handling**: Use `anyhow::Result` with `.context()`/`.with_context()` for model loading and frame I/O so failures name the file or stage.
- **Real-Time Constraints**: Keep `process_frame()` allocation-light; avoid blocking I/O (file writes, MIDI sends) inside the cpal audio callback — that callback must stay sample-rate safe.

### 2. Coding Style
- Follow standard Rust idioms (`rustfmt`/`clippy`), prefer explicit `?` over `.unwrap()`/`.expect()` in vision/audio paths.
- Treat the existing `unsafe impl Send for StreamWrapper` and `unsafe { ... }` pointer sampling as deliberate; add a comment explaining safety before touching them.

### 3. Never do these actions
- NEVER run `ls -R` — `target/` holds ~1.7 GB of dependency artifacts.
