# Changelog

Notes by development phase. Dates are the merge dates in the history; the branch names
in parentheses are the branches each phase was developed on.

## Unreleased

- **OpenCV is gone.** `opencv` 0.102 was the project's only native dependency and it was
  doing three chores: grabbing pixels, cropping/resampling into the model input, and
  drawing the `--image` overlay. All three are now Rust:
  - `src/core/frame.rs` - a `Frame` of RGB, row-major, replacing `opencv::Mat` throughout.
  - `src/vision/sampling.rs` - a hand-written bilinear `sample_affine_to_tensor` that
    writes the NHWC/NCHW float tensor in one pass (the pipeline's mappings are all
    axis-aligned, so `warpAffine` was only ever reached when a crop hung off the edge).
  - `src/presentation/ascii.rs` - a box-average downsample, which is what OpenCV's
    `INTER_AREA` was doing.
  - `src/presentation/annotator.rs` - Bresenham lines and filled discs instead of
    `imgproc::line`/`circle`.
  - `src/vision/camera.rs` - webcam capture via `nokhwa` (AVFoundation) with a
    hand-written YUYV -> RGB pass; nokhwa's own decoder takes 53 ms per 640x480 frame,
    this one takes 1.4 ms.
  - `src/vision/video.rs` - MP4/H.264 decoding via `mp4io` + `rusty_h264-decoder`, both
    `#![forbid(unsafe_code)]`. Verified bit-exact against ffmpeg on the fixture; the RGB
    conversion lands within 2 LSB of ffmpeg's.
- Cold `cargo build --release` drops from ~19 minutes to ~4 minutes, and `.cargo/config.toml`
  no longer needs to point `LIBCLANG_PATH` at Homebrew LLVM.
- `build.rs` links clang's compiler-rt on Apple targets: fat LTO drops
  `___isPlatformVersionAtLeast` from `std`, which ONNX Runtime's CoreML object references
  (rustc issue #164915). Without it `cargo build --release` fails to link on macOS.
- ~~`fixtures/hand_clip.mp4`~~ and `fixtures/clip_trace.json` are **not committed**: the clip is
  a recording of a real hand, so `make record` generates it locally and `make trace` regenerates
  the golden trace against it. `make fixture|trace-check|trace` say so when either is missing.
- `--video` decodes the clip with `mp4io` + `rusty_h264-decoder` (both `#![forbid(unsafe_code)]`).
  `rusty_h264-decoder` is 4:2:0-only, hence the `-pix_fmt yuv420p` in `make record`.
- **CI has a regression gate it can actually run.** There is no macOS runner, so CI can never
  touch the camera or the gitignored ONNX assets. Mock mode is driven by the frame index, so
  `fixtures/mock_trace.json` (120 frames, deterministic across debug and release builds) is
  committed and CI diffs against it — `make mock-trace-check`.
- `src/midi/mod.rs` maps `midir::ConnectError` to a message before the `?`. On the ALSA backend
  that error carries a raw pointer, so it is not `Sync` and cannot become an `anyhow::Error`;
  the CI build failed with E0277 while macOS built fine.
- The detector **box is off by default** in the dashboard (`show_box: false` in `Dashboard::new`).
  The `box` header button still turns it on; it just starts hidden so the stage opens as the
  keyboard plus skeleton rather than a rectangle around the hand.
- **Fix: the stage no longer panics before the first frame is published.** The keyboard glow is
  a per-note buffer, but `main` seeded the snapshot with an empty one and `Pipeline::pause`
  cleared it, while the layout the stage draws is always full. Whichever side won the race -
  a UI tick landing before the frame loop's first publish, or a Stop press - the first draw
  indexed past the end of the trail (`index out of bounds: the len is 0 but the index is 36`).
  `SoundView::default()` now carries one level per note, `pause` zeroes the buffer instead of
  emptying it, and `keyboard_trail` reads levels through a clamped getter.

## 0.1.0 — 2026-11-15 · standalone crate

- Renamed the crate and repo to **momotone**; the Rust engine left the Python PoC repo
  and became its own repository, with all 16 branches and the full history.
- Dependencies: `cpal` 0.17 → 0.18 (`build_output_stream` now takes `StreamConfig` by
  value), `opencv` 0.100 → 0.102, `clap` 4.6.6 → 4.6.7, `serde_json` 1.0.151 → 1.0.154,
  `tokio` 1.53.1 → 1.54.1, `ort` rc.13 → rc.15, `crossbeam` 0.5 → 0.5.17, `midir` 0.11 → 0.11.1.
- Toolchain: rustc 1.98 → 1.100, Homebrew LLVM 23.1.5 for the OpenCV bindings.
- `docs/blazehand-model-notes.md` — the pasted BlazeHand model readme moved out of the
  project README.

## Phase 5 · observability — 2026-11-13/14 (`phase5-observability`)

- `tracing` everywhere: one `EnvFilter` at the subscriber, `--debug` is a level filter,
  `RUST_LOG` overrides it; no more `println!` or per-module debug flags.
- Per-stage percentiles (p50/p95/p99/max) on the dashboard Performance card, with an
  over-budget marker (`--budget-ms`, default 33 ms).
- Budget guard: the pipeline sheds `feed` and `ascii` itself when inference's p95
  blows the frame interval.
- Reproducibility: `--video` fixture source, `--trace` / `--expect` golden traces,
  `fixtures/hand_clip.mp4` + `fixtures/clip_trace.json`, and `make fixture|trace-check|trace`.
- Telemetry rings moved into `core/telemetry`; gauge geometry (meters, sparklines, the
  NaN guard) made pure and tested.

## Phase 4 · snapshot & presentation — 2026-11-11/12 (`phase4-snapshot`)

- `FrameMetrics` split into five views with one writer each: hand, sound, perf, scene, status.
- Presentation layer on its own thread: RGBA conversion, ASCII render, stdout and
  `--ascii-dump` never run on the frame loop; the frame is *moved* into a capacity-1 job.
- Keyboard geometry is pure data (`Keyboard::new`), testable without iced; backdrop and
  grid are cached canvas layers.
- The keyboard trail is pipeline state, not dashboard state — the held cell glows,
  released cells decay with `--fade-ms`.

## Phase 3 · instrument core — 2026-11-11 (`phase3-instrument`)

- `HandEstimator` — one source of truth for the hand: confirm/grace gate, fingertip EMA,
  proximity EMA, geometry, jitter. Strategies read it and mutate nothing.
- `Performance { Note, Signal, Release }` replaced the old event vocabulary.
- Lock-free audio: the cpal callback owns one `Voice`, reads four atomics, allocates
  nothing and makes no syscall (12 ms attack, 60 ms release, 15 ms pitch glide).

## Phase 2 · pipeline structure — 2026-11-10 (`phase2-*`)

- One `RuntimeConfig` and one `UiCommand` channel; the dashboard never writes pipeline
  state directly.
- One frame path (the mock path and the live path were the same code); `advance()`
  rewritten as five named stages, `Pipeline`'s fields grouped.
- `DetectionGate` confirm/grace debounce — the anchor score flickers around the
  threshold, so a raw compare would click the note on and off.
- Makefile gate (`make verify`) with clippy warnings as errors.

## Phase 1 · hot path — 2026-11-10 (`phase1-hot-path`)

- Zero-copy `StageInput` buffers handed to ort as `TensorRef` views; OpenCV
  `resize`/`warpAffine` resampling instead of manual pixel loops.
- Landmark reuse while the palm is stationary (`ReuseCache`, ≤3 frames in a row).
- Stopped cloning landmarks per frame; RGBA conversion gated on the `feed` toggle and
  `--feed-fps`; mirror flip writes into a reused scratch `Mat`.
- LTO + single codegen unit release profile.

## Phase 0 · baseline — 2026-11-10 (`phase0-telemetry-baseline`)

- Measured per-stage timing percentiles before touching anything, recorded in
  `bench/baseline.md`.
- All clippy lints cleared; architecture review and the sequenced remediation plan
  written (`ARCHITECTURE.md`, `MASTER_PLAN.md`).

## 2026-10-25 → 10-29 · the instrument takes shape

- ONNX two-stage hand pipeline in Rust (`ort`) with configurable MIDI note range.
- iced HUD replacing the OpenCV overlay; gradient stage, camera feed in the rail.
- Hand-following detector ROI (`min(W,H)` square sliding to the last palm centre) —
  the centred window made the outer keys physically unreachable.
- Degree × octave grid keyboard (`GridLayout`, `NOTE_MARGIN` band), pitch from fingertip x.
- Scale quantisation (`--scale`, `--root`, `--octaves`), instrument pickers in the rail.
- MIDI note gate (`--min-note-ms`) and the theremin strategy.
- Single-pass Ultralytics pose tracker behind `--model yolo`, layout-agnostic decoder
  (YOLOv8/v11 attribute-major and YOLO26 detection-major heads).
- ASCII luminance backdrop tinted by the note being played.
- Capture, inference and sinks moved to their own threads.

## 2026-08-24 → 08-27 · scaffolding

- First Rust skeleton: camera source, tracker, mock source, strategies, visualizer.
- `midir` 0.11 upgrade and the first working MIDI output.
