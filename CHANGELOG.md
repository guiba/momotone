# Changelog

Notes by development phase. Dates are the merge dates in the history; the branch names
in parentheses are the branches each phase was developed on.

## v0.2.0 — 2026-11-25

- **Unsigned DMG release.** `make release` (and `.github/workflows/release.yml` on a
  `v*` tag push, running `scripts/release.sh` on a self-hosted macOS runner) builds an
  unsigned `Momotone.dmg` with the models bundled inside the `.app`. The binary now falls
  back to `Resources/assets/` inside the bundle when the default asset paths do not resolve
  relative to the cwd, so the installed app finds its models. Releases are unsigned and will
  be blocked by Gatekeeper on first run until a Developer ID is set up.

- **Two hands.** `--hands one|two|probe` (default `two`) turns the instrument into a
  pitch hand plus a modulator hand. The pitch hand still comes from the palm-anchored
  192×192 crop - the more sensitive window - and a second letterboxed whole-frame pass
  finds the modulator, which the crop physically cannot reach. Both hands run the same
  landmarker with their own reuse cache, so the second hand is detected exactly like the
  first. `HandDetectionResult` now carries `hands: Vec<HandDetection>` in role order with
  `MAX_HANDS = 2`, and `HandEstimator` is instantiated twice. Palm anchors above the score
  gate are merged by greedy spatial NMS (`non_overlapping`) at the measured
  `HAND_SEPARATION = 1.0`, and hands keep their slot across frames (`match_to_slots`,
  `SLOT_MAX_JUMP = 0.35`) so the two hands do not trade identities when the detector's
  scores flip. The modulator drives the effect chain - its proximity, height and horizontal
  position - and the distance between the palms drives vibrato depth and echo feedback; with
  one hand in view that hand drives everything, exactly as before. The gap is normalised by
  the hands' own sizes (`gap_level(distance, touching)`), so it reads the same at 640×480
  and 1620×1080. The dashboard gains a HANDS card with a **swap hands** button, and the
  modulator skeleton is drawn in a cooler, dimmer palette labelled "mod". Measured on the
  two-hand fixture: inference p50 16.9 ms with two detector passes and two landmarker runs,
  against 11.8 ms with one hand - still inside the 33 ms frame budget.
- **A detector-window probe.** `--hands probe` runs the anchored crop and the whole-frame
  pass over the same frames and prints the comparison - detections, mean score, hands per
  frame at each NMS separation, and µs per detector pass. `make probe`, `make probe-single`
  and `make probe-640` run it on the fixtures. It is a measuring mode, not a playing mode.
  The measurement it produced is what settled the two-hand design (see `SECOND_HAND_PLAN.md`
  §16).
- **An effect chain, driven by the hand.** `src/core/effects.rs` declares the stages
  (`lowpass`, `vibrato`, `delay`) as source -> parameter -> curve, and the frame loop
  smooths their parameters every frame and puts them on the bus as
  `BusMessage::Modulation`, which sinks receive through `Sink::set_modulation` (a
  default no-op, so the MIDI engine is unaffected for now). The synthesizer runs
  `glide -> vibrato -> oscillator -> one-pole lowpass -> comb delay -> soft clip`; the
  delay line is allocated once when the stream opens, and the audio callback still takes
  no lock and allocates nothing. `--effects lowpass,vibrato,delay|none` sets the chain
  at launch and the dashboard's EFFECTS card toggles it live. When the hand leaves the
  frame the chain fades to neutral over `--fade-ms` rather than switching off, and a
  released note rings out through an echo that is already in the ring instead of being
  cut off. Measured cost: ~7 ns per audio sample, ~0.2 ms of the 33 ms frame budget, and
  it is not spent in the frame loop at all.
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
- **A quiet start.** `init_logging` caps `iced_winit`, `iced_wgpu` and `ort::logging` at WARN.
  Opening the dashboard used to print a dozen lines of library internals - the window
  attributes struct, the wgpu compositor settings, the adapter and format lists. A run
  now starts with the seven lines that describe *this* run; ONNX Runtime's single CoreML
  capability warning still shows, and `RUST_LOG=info,iced_winit=info` gets the detail back.
- **The echo no longer crackles.** Two bugs in the delay stage made it sound like noise
  rather than an echo. `delay_ms` is the fingertip's horizontal position, so it wobbles
  constantly, and the comb tap was an integer: every time the modulation crossed the
  2 ms deadband the tap jumped by up to 88 samples, i.e. the feedback loop read an
  unrelated sample of the ring - an impulse. Across one measured sweep that is a click
  every 256 samples (worst sample-to-sample step 0.69 against the note's own slope of
  0.025). The tap is now a float that slews at most one sample per audio sample
  (`DELAY_SLEW`), read by linear interpolation between the two samples either side of it.
  Second, a feedback comb has a loop gain of `1/(1 - feedback)` at its resonances, so 0.6
  feedback made a note 2.5x louder than the envelope asked for and parked it in the soft
  clipper: ~24 % harmonic distortion. The stage is now pre-scaled by `1 - feedback`, so a
  held note comes out at the level the hand set and the tail still decays by `feedback`
  per repeat. Two new regression tests fail on the old behaviour. A fresh attack after the
  voice went silent empties the ring and snaps the float tap to the current echo time, so
  the previous note's echo cannot layer onto the attack and the tap cannot chirp through
  a slew while the hand sat still; the tail on a release is unchanged.
- **The lowpass now bites.** Its knee was an absolute `200-8000 Hz`, which for the
  notes this instrument plays (C2-C6, 65-1047 Hz) sits above every fundamental: a
  pole at 4 kHz attenuates a 440 Hz sine by 1 %, so switching the stage on or off
  was inaudible except when the hand backed right off. The knee is now measured in
  harmonics of the sounding note (`CUTOFF_MIN_HARMONICS = 0.5` to `CUTOFF_MAX_HARMONICS
  = 3.0`, `cutoff_for(frequency, proximity)`), and it is two cascaded poles instead
  of one - 12 dB per octave rather than 6. At the dark end a note loses ~14 dB and
  a square loses its harmonics down to the second; the EFFECTS card shows the knee as
  `2.4x harmonics (627 Hz)` instead of a bare hertz value.

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
