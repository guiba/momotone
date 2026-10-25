# AGENTS.md - AI Assistant Context for gesture_rust

This document provides structural and architectural context to guide AI assistants in working with this codebase.

## Project Overview
`gesture_rust` is a high-performance, real-time hand gesture recognition engine implemented in Rust. It processes webcam input via OpenCV, extracts hand landmarks, and translates those movements into audio signals (via `cpal`) or MIDI messages (via `midir`) using an asynchronous event-driven architecture.

## Tech Stack
- **Language**: Rust (Edition 2021)
- **Vision**: `opencv` (OpenCV bindings), `ndarray` (Linear algebra/matrix ops)
- **Audio**: `cpal` (Low-level audio I/O)
- **MIDI**: `midir` (MIDI communication)
- **Concurrency**: `tokio` (Async runtime), `crossbeam-channel` (High-performance MPMC channels)
- **CLI/Config**: `clap` (CLI parsing), `serde`/`serde_json` (Serialization)
- **Error Handling**: `anyhow`

## Core Architecture & Data Flow
The project follows a **Producer-Consumer** pattern mediated by an **Event Bus**.

1.  **Input Phase (Sources)**:
    *   `src/vision/camera.rs`: Captures frames from the webcam via OpenCV.
    *   `src/sources/mock.rs`: Provides mock data for testing without hardware.
    *   Both implement traits defined in `src/core/mod.rs` (see `Source`).

2.  **Processing Phase (Vision)**:
    *   `src/vision/tracker.rs`: Uses hand detection models to identify landmarks within video frames. 
    *   Results are encapsulated in `HandDetectionResult`.

3.  **Orchestration Phase (The Glue)**:
    *   `src/core/event_bus.rs`: The central hub (`EventBus`) that routes `GestureEvent`s based on their `EventType`.
    *   `src/core/types.rs`: Defines the core data models like `Landmark`, `GesturePayload`, and `EventType`.

4.  **Output Phase (Sinks)**:
    *   Subscribers to the Event Bus implement the `Sink` trait:
        *   `src/audio/synthesizer.rs`: Converts gestures into continuous audio tones.
        *   `src/midi/mod.rs`: Maps gestures to MIDI notes/CC messages for external hardware (e.g., OP-1).
        *   `src/visualizer/mod.rs`: Renders the detection overlays onto the video feed for UI feedback.

## Key Module Reference
| Module | Responsibility |
| :--- | :--- |
| `src/main.rs` | Entry point, CLI argument parsing (`Args`), and lifecycle management. |
| `src/core/` | The backbone: Traits (`Source`, `Sink`, `Tracker`) and shared data types. |
| `src/vision/` | Computer vision implementation (Webcam capture and Landmark tracking). |
| `src/audio/` | Audio synthesis engine and stream management. |
| `src/midi/` | MIDI communication protocols and output mapping. |

## Development Guidelines for AI Assistants

### 1. Design Patterns
- **Trait-Based Abstraction**: When adding a new input (e.g., a different camera API) or a new output (e.g., OSC), implement the `Source` or `Sink` traits from `src/core/mod.rs`.
- **Asynchronous Runtime**: The project relies on `tokio`. Ensure long-running tasks or hardware I/O are handled without blocking the main event loop.
- **Error Handling**: Use `anyhow::Result` for all functions that can fail, particularly in the vision and audio pipelines.

### 2. Real-Time Constraints
- This is a real-time application. Avoid heavy computational tasks or blocking I/O inside the primary detection or audio callback loops.
- Prefer `crossbeam-channel` for high-throughput message passing between the vision thread and the output engines.

### 3. Coding Style
- Follow standard Rust idioms (`rustfmt`/`clippy`).
- Use explicit error propagation (`?`) rather than `.unwrap()` or `.expect()` in production-critical code (audio/vision).

### 4. Never do this actions
- NEVER run ls -R there are a lot of external dependency files i target folder
- NEVER mixes find/replace and line-range fields when using edit tool
|
