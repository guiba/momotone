# Roadmap — momotone

Effort tags: **S** (a few hours), **M** (a day-ish), **L** (multi-day). Ordered by value-to-effort.
Status: measured on `master` after Phases 1–5 of `MASTER_PLAN.md`. Items struck through are done;
the numbers below are from the current tree.

## Measured baseline

| What | Number | Where |
| :--- | :--- | :--- |
| Frame loop (MediaPipe, live camera) | **29.9 fps**, 33.4 ms/frame — capture 33.4 ms, inference 10.2 ms p50 (p95 11.7), 0 dropped | `make headless` |
| Frame loop (fixture clip, headless) | **29.7 fps**, 33.34 ms/frame — capture 0.2 ms, inference 11.1 ms p50, 0 dropped | `make fixture` |
| Frame loop (pose model) | **21 fps**, 41 ms/frame — capture 29 ms, inference 69 ms, 13 of 73 dropped | same with `--model yolo --score-threshold 0.3` |
| Camera read | ~31 ms; the loop is paced by the driver, not by inference | end-of-run `INFO` summary |
| Detector stage (192×192, hand absent) | 9.5 ms | `--debug` |
| Detector input sampling (192×192) | **146 µs** (reused buffer), 162 µs fresh — hand-written bilinear, slightly slower than OpenCV's vectorised `resize` (133 µs) | `cargo test --release -- --include-ignored bench_sampling --nocapture` |
| Landmarker input sampling (224×224) | **254 µs** vs 278 µs fresh | same bench |
| RGBA conversion (`frame_to_rgba`) | 0.30 ms/frame — only when the feed is on *and* due | presenter benches |
| Mirroring | **0.018 ms/frame**, no allocation after the first frame | `cargo test --release -- --include-ignored bench_mirror_flip` |
| Mock-mode clock | **33 ms/frame** — a deadline clock, not a flat sleep | `--mode mock --no-ui --max-frames 120 --debug` |
| rustc warnings | 0 (only the third-party `block v0.1.6` future-incompat notice) | `cargo build` |
| `unsafe` in the crate | 1 — `unsafe impl Send for StreamWrapper` in `audio/synthesizer.rs` | `cargo` + review |
| Cold `cargo check` | **34 s** (was 19 m 12 s with OpenCV) | `cargo clean && cargo check --all-targets` |
| clippy lints | 0 — `make clippy` is `-D warnings` and part of `make verify` | `make verify` |
| Dead code | 1 flag, `Visualizer::draw_skeleton`, which *is* used by `--image` — a call-graph false positive | `aft_inspect` |

## Codebase census (this sweep)

- **37 source files, 12,178 lines.** `pipeline.rs` and `dashboard/skeleton.rs` are the two
  big ones; the iced layer is ~1,950 lines.
- **181 tests pass, 7 benches are `#[ignore]`d.** Coverage now spans the strategies, the
  estimator, the detection gate, the affine sampler, the ROI window, both pose-output
  layouts, the geometry helpers, the mock hand, the release and pause paths, the MIDI note
  gate, the trail decay, the synthesizer voice lifecycle, the presenter cadence, the
  keyboard band/cell geometry, the trace differ, the video pacing and B-frame reordering,
  the `Frame` invariants, the ASCII ramps and the pure-Rust skeleton overlay.
- **Still zero tests** in `dashboard/mod.rs`, `dashboard/gauges.rs` and `main.rs`. The keyboard
  geometry (T4.5) proved the pattern — split the arithmetic from the widget and it becomes testable;
  `gauges.rs` and `metrics.rs` are next.
- **Duplicate groups:** three ~20-line pairs in `vision/tracker.rs` — all inside `bench_sampling`, i.e. benchmark scaffolding, not production duplication.
- **Import cycles:** 0.

## New findings worth scheduling

1. ~~Mock mode runs at 25 fps, not 30~~ **done** — a deadline clock (`Telemetry::next_mock`) aims
   at the 33 ms interval and sleeps only the remainder, so mock telemetry is honest. The mock hand is
   also driven by the *frame index*, which is what makes its trace a golden reference.
2. ~~The MIDI sink re-sends `note-on` every frame for a held note~~ **done** — the sink carries a `NoteGate`, a held note attacks once, volume travels as CC 7, and `--min-note-ms` quantises a fast sweep into an arpeggio.
3. ~~The oscillator is not band-limited and frequency changes are not phase-continuous~~ **done** — `src/audio/synthesizer.rs` now uses the device sample rate, BLIT-style band-limited square/saw capped at `MAX_HARMONICS = 24`, soft clipping, a 12 ms attack, a 60 ms release and a 15 ms pitch glide.
4. ~~`ort` sessions are created with default settings~~ **mostly done** — `--ort-threads` pins ORT intra-op threads (0 = runtime default, which measured fastest here). Pinning through the rc API hit trait-bound friction in `ort 2.0.0-rc.13`; the knob exists, the measurement is still open.
5. ~~Vestigial payload: nobody emits `GesturePayload::Signal`~~ **done** — it is now real: `--gesture theremin` (`src/gesture/theremin.rs`) maps the index tip's x to a logarithmic C2..C6 band and the thumb-index spread to amplitude. `EventType`/`GestureEvent.event_type`, which nothing ever read, were deleted, and `GestureEvent`/`GesturePayload` became the `Performance { Note, Signal, Release }` enum.
6. ~~Dead fields in the shared snapshot~~ **done** — `FrameMetrics.frame_id` and `GestureEvent.event_type` are gone, and `FrameMetrics` is now four named views (`hand`, `sound`, `perf`, `scene`, `status`) with one writer each.
7. ~~Doc drift~~ **done** — `AGENTS.md` now says `DEFAULT_SCORE_THRESHOLD = 0.6` (the code's value), drops the phantom `ndarray 0.17` dependency, describes the deleted scaffolding as deleted, and lists `--gesture` / `--theremin-low-hz` / `--theremin-high-hz` / `--min-note-ms`.
8. ~~No detector hysteresis~~ **done** — `src/core/detection_gate.rs` requires 2 confirm frames and grants a 3-frame grace window, so the flickering anchor score no longer clicks the note on and off.
9. **Camera resolution is larger than the model ever sees** — capture is 640×480 but the detector samples a 192×192 square and the landmarker a 224×224 one. Try `--camera-width 320 --camera-height 240` and compare `capture` in the end-of-run summary; the camera read (32 ms) *is* the loop's pacing, so cheaper pixels are the only remaining lever on end-to-end latency. **S**
10. ~~The UI's only unshared state is the trail~~ **done** — `KeyboardTrail` is owned by the pipeline and published in `SoundView`, so the stage draws a fact about the run rather than maintaining its own history. The keyboard's cell geometry is a plain `Keyboard` struct, testable without iced.
11. ~~No CI, no clippy gate~~ **done** — there is no GitHub Actions by design; `make verify` is the gate (`check` + `test` + `fmt --check` + `clippy --all-targets -D warnings`) and clippy is clean.
12. ~~Latency is reported as an EMA only~~ **done** — `Samples`/`StageProfile` keep rolling p50/p95/p99/max for capture, inference, camera→publish and dispatch→sink, printed at `INFO` at the end of a run and shown on the Performance card with an over-budget marker.

## Now — highest value next

1. ~~MIDI note-on/off discipline~~ (finding 2) + **retrigger on cell change** — one attack per cell transition, with a `--min-note-ms` floor so a fast sweep becomes an arpeggio rather than a blur. **done for MIDI; the audio sink glides by design** (see finding 3).
2. ~~Mock-mode deadline clock~~ (finding 1) **done**.
3. ~~Tracker fixture tests~~ **done** — `tracker_finds_the_hand_in_the_clip` runs the real ONNX graphs over `fixtures/hand_clip.mp4` (106/120 detections, all 21 landmarks).
4. ~~CI~~ (finding 11) **done** — `make verify`.
5. ~~Audio: band-limited waveforms + ADSR + portamento~~ (finding 3) **done**.
6. ~~Detector hysteresis~~ (finding 8) **done**.
7. ~~ORT thread pinning~~ (finding 4) **partly done** — `--ort-threads` exists; the measurement is still open.
8. ~~Cache the keyboard backdrop~~ **done** — `StageCache` in `dashboard/skeleton.rs` caches the backdrop and the grid; only the trail, play head, feed and skeleton are redrawn each tick.
9. ~~Housekeeping sweep~~ (findings 5, 6, 7) **done** — see "Recently done".
10. ~~Kill the per-frame clones~~ **done, but it was not the bottleneck** — the tracker now moves the landmark Vec into the result instead of cloning it. Measured: copying the whole `FrameMetrics` snapshot costs **0.44 µs/frame** (0.0015 % of a 33 ms frame), so a hand-written `copy_from` was not worth the maintenance risk.
11. ~~Pitch hysteresis / smoothing~~ **done** — the index tip runs through an EMA in the estimator before quantisation.
12. ~~Release profile + per-stage latency printout~~ **done** — `lto = "fat"`, `codegen-units = 1`, and the stage tails are p50/p95/p99/max.
13. ~~Clear the 13 dead-code warnings~~ **done** — deleted `sources/mock.rs`, `SimplePinchStrategy`, `FrequencyStrategy`, `Visualizer::draw_hud`, the core `Source`/`Tracker` traits, `GestureEvent::new_signal` and `MidiOutputEngine::control_change`; `StreamWrapper`'s unread field is now `#[allow(dead_code)]` with a comment (holding the stream *is* the point).

## Model backends (`feat/yolo-pose`)

The tracker is now an enum, `HandTracker::MediaPipe` / `HandTracker::YoloPose`, selected with `--model`. Both return the same 21 MediaPipe-ordered landmarks plus a square hand box, so the strategy, dashboard and visualizer are unchanged.

Measured on this machine (release build, one frame of a 640 px hand photo, `--image`):

| Backend | Models | Input | Score | Inference |
| :--- | :--- | :--- | :--- | :--- |
| `mediapipe` (default) | 4.6 MB + 10.9 MB | 192×192 + 224×224 | 0.965 | **11.6 ms** (7.3 + 4.3) |
| `yolo` | 41 MB FP32 (21 MB FP16) | 640×640, static | 0.481 | **64 ms** |

So: the Ultralytics pose model is a single pass and gets the whole hand in one shot (no palm-box guessing), but at 640 it is ~5× slower than the BlazeHand pair on a CPU and its confidences are much lower — start at `--score-threshold 0.3`. `scripts/get_pose_model.sh` fetches the graph. Open items:

1. **Make the pose model cheap** — the HF export is frozen at 640×640. Re-export with `yolo export --imgsz 320 --dynamic` (needs torch/ultralytics, not installed here) or find a nano export; at 320 the FLOPs drop ~4×. **M**
2. **Try RTMPose** — RTMDet-nano (76 AP, 0.31 GFLOPs, 320×320) + RTMPose-m hand5 (256×256) is the accuracy/latency sweet spot and already downloads to `/tmp/handbench`. Its SimCC heatmap head needs a different decoder than `decode_pose`. **M**
3. **Multi-hand** — the pose decoder keeps only the highest-scoring detection; the strategy and dashboard assume one hand. **M**

## Performance

- ~~Skip the landmarker when the hand is static~~ **done** — `ReuseCache` reuses the previous keypoints while the palm is within 3 px and 3 % of its last size, capped at 3 consecutive frames so a hand that reshapes without moving its palm is never served stale landmarks for long, and cleared whenever the detector loses the hand. The landmarker stage costs ~4-5.4 ms when it runs, so a still hand saves most of it. Verify with `cargo run -- --debug` (`palm steady, reusing landmarks`) and compare against `--landmarker-always`. **Note:** fps is camera-paced at ~30, so the win is latency and CPU headroom, not frame rate.
- ~~Mirror flip in place~~ **done, but it was never a bottleneck** — the bindings take two separate borrows, so `flip(frame, frame, 1)` is not expressible; `WebcamSource` now carries a scratch `Mat` and swaps it with the captured frame, so the destination allocation happens once instead of every frame. Measured 0.0176 ms/frame (fresh Mat) vs 0.0174 ms/frame (reused scratch): the allocator recycles the 920 KB block, so the saving is ~0.0002 ms/frame. Flipping the landmarks' x instead of the pixels would skip the memcpy too, but that is at most 0.018 ms of a 31 ms frame — inference and the camera read are where the time is. Verified by `mirroring_reverses_the_columns` and `a_second_flip_reuses_the_scratch_buffer_instead_of_allocating`.
- ~~Split capture, inference and sinks onto their own threads~~ **done** — `Pipeline::run` now runs three stages: the capture thread reads the webcam into a capacity-1 channel (a frame inference has not reached yet is dropped and counted, so inference never works on stale pixels), the pipeline thread does tracker → strategy → publish, and a sink thread owns the event receiver and drives the audio/MIDI outputs. `frame_ms` in `FrameMetrics` is now inference time only; `capture_ms`, `latency_ms`, `captured` and `dropped` sit next to it and the Advanced tab's Performance card shows them. `fps` is measured from the gap between published frames rather than the sum of the stages, since the stages now overlap. `Pipeline::tick` remains the single-frame inline path (mock mode and the tests use it). Run with `cargo run --release -- --debug --no-ui` and compare `inference` against `capture`: the loop is paced by whichever stage is slower.
- ~~Trim `tokio`~~ **done** — `Cargo.toml` now declares `features = ["rt", "time"]` instead of `["full"]`.
- ~~Audio callback lock~~ **done** — the cpal callback reads four atomics (`Target { sounding, frequency, amplitude, waveform }`) and owns a single `Voice`; it takes no lock, allocates nothing and makes no syscall.
- **Camera: capture at 320×240** (finding 9) — the detector only ever sees a 192×192 square. **S**
- **ORT session tuning** (finding 4). **S**
- **GPU / alternative ONNX Runtime execution provider** for `ort`. **L**
- **Single-shot hand model** instead of detector + landmarker, or a retrained lighter BlazeHand variant. **L**
- ~~Pin capture FPS~~ **done** — `WebcamSource::new` sets `CAP_PROP_FPS` (`--camera-fps`, default 30) and `--debug` prints what was requested versus what the driver reports, so a camera defaulting to 60 fps no longer doubles the inference load unnoticed.

## Features

### Instruments & layouts
- ~~Quantise pitch to a scale~~ **done** — `src/core/scale.rs` holds seven presets (major/minor pentatonic, major, minor, dorian, whole tone, chromatic) plus root-name parsing; `--scale` / `--root` choose it. Every reachable cell is consonant with every other one.
- ~~Degree × octave matrix~~ **done** — `src/core/grid.rs` builds a `GridLayout` (columns = scale degree, rows = octave, top row highest) centred inside C2–C6 and shrunk if the requested span cannot fit. `FingerPositionStrategy` EMA-smooths the index tip in **both** axes and maps it to a cell; the canvas draws the same cells with the same `NOTE_MARGIN` inset, labels octave rows, and names each cell when it is big enough. `--octaves` sets the row count (default 3).
- ~~Dashboard pickers~~ **done** — an `INSTRUMENT` card at the top of the basic rail with pick lists for scale, root, octave span and waveform. They write into one `Arc<Mutex<InstrumentConfig>>` (`src/core/instrument.rs`); the pipeline compares its `version` at the top of every tick and, only when it moved, re-keys the strategy and pushes the new timbre to the sinks. No restart needed.
- **Retrigger on cell change** — one attack per cell transition instead of a continuous re-tune, with a `--min-note-ms` floor so a fast sweep becomes an arpeggio rather than a blur. Pairs with finding 2. **S**
- ~~Waveform + ADSR in `Voice`~~ **partly done** — sine/triangle/square/saw are band-limited, there is a 12 ms attack and a 60 ms release, and pitch glides over 15 ms. Detuned pairs and a full ADSR are still open. **M**
- **Vertical band calibration** — learn the y range the user actually occupies from recent landmark bounding boxes, so the top row is not unreachable. **M**
- **Named instrument presets** — bundle scale + root + octave span + waveform + envelope (e.g. *Kalimba*: C pentatonic, pluck). **S**
- **Velocity curve** — volume is currently a linear read of thumb↔index spread saturating at 60 % of the hand box. An exponent or a per-user curve (see `--calibrate`) would make the instrument feel less like a gate. **S**

### Gesture layer
- **Landmark smoothing (EMA / Kalman-lite)** before any strategy runs — the tip is smoothed, the other 20 joints are not, so `spread`, `extension` and `tilt` are noisier than the pitch. **S**
- ~~Detection hysteresis~~ (finding 8) **done** — `DetectionGate` debounces the confidence gate with 2 confirm frames and a 3-frame grace window.
- **Discrete gesture set**: open hand, fist, pinch, peace, with per-gesture mapping configured rather than hardcoded. **M**
- **Finger extension as switches** — `hand_geometry()` already computes per-finger extension; turn each into an on/off gate. Concretely: thumb-down = octave up, index+middle = sustain, pinky = mute. **M**
- **Gesture chords** — combine several gestures into chord/scale selection. **M**
- **Use `Landmark.z`** — depth is parsed but read by nothing; it is the cheapest way to distinguish "hand near the camera" from "hand far away" for volume. **M**
- ~~Continuous/theremin strategy~~ **done** (finding 5) — `ThereminStrategy` emits the `Signal` payload the sinks already handled.
- **Two hands** — NMS over top-k anchors instead of a single argmax, `Vec<HandDetectionResult>`, left/right roles (pitch vs modulation). **L**

### Output sinks
- **OSC sink** — implement `Sink` and push it into `sinks` in `main.rs`. **S**
- **MIDI polyphony** — `MidiOutputEngine` is monophonic (`held_note: Option<u8>`); add a voice pool. **M**
- **JSON/CSV event log** for offline analysis of gesture traces — `serde`/`serde_json` are already dependencies and `GestureEvent`/`GesturePayload` already derive `Serialize`, so this is a writer plus a `--log <path>` flag. **S**
- ~~MIDI velocity as CC~~ (finding 2) **done** — a held note's loudness is CC 7; note-on velocity is only sent at the attack.

### Input & evaluation
- ~~`--video <file>` playback source~~ **done** — `VideoSource` reads a clip paced at its own frame rate, so a run is reproducible and `make trace-check` can diff it against a golden trace.
- ~~Benchmark harness~~ **done** — `fixtures/hand_clip.mp4` + `bench/baseline.md` + `make fixture`/`make stats`.
- **`--calibrate`** — sample the user's comfortable hand range and store per-user min/max, replacing the fixed `NOTE_MARGIN`. **M**
- ~~Stage latency percentiles~~ (finding 12) **done** — p50/p95/p99/max per stage on the Performance card.

### Visualisation & UX
- **Keyboard shortcuts** for Start/Stop and the toggles (iced key events) so the app is operable without hunting for buttons. **S**
- **Stage overlay while paused** — a "stopped" banner on the keyboard so the state is obvious from a glance, not just in the Status card. `Skeleton` already receives `lost_ms` but not `paused`. **S**
- **In-UI fade control** — expose `--fade-ms` as a slider/stepper in the Advanced tab. **S**
- **Show `--feed-fps` in the header** — the preview cadence is currently invisible, so a 15 fps feed reads as a bug rather than a setting. **S**
- **Note-name labels on the keyboard** (currently only the octave Cs); highlight the current pitch class. **S**
- **Audio visualiser** — render the oscillator envelope or an RMS trace from the cpal callback. **M**
- **Voice count / MIDI note-on-off state** in the rail. **S**

### Tooling
- ~~CI~~ (finding 11) **done** — `make verify` runs `check`, `test`, `fmt --check` and `clippy --all-targets -D warnings`. No hosted CI by design.
- ~~Tracker fixture tests~~ **done** — see `tracker_finds_the_hand_in_the_clip`.
- **UI unit tests** — `Mapping`/letterbox math, `fade_trail` and the note-name tables are pure functions with little test today. The keyboard geometry (T4.5) is the template: a plain struct, no iced. **S**
- ~~Replace `println!`/`eprintln!` with `tracing`~~ **done** — one `EnvFilter` in `main.rs`; `--debug` sets the level, `RUST_LOG` overrides it, and `ort::logging` is pinned to `warn`. No `debug: bool` field survives in any struct.
- **TOML config** for model paths, thresholds, note ranges, with CLI overrides. **M**
- ~~Split the complexity hotspots~~ **partly done** — `advance()` is five named stages and `Skeleton::draw` is split into cached backdrop/grid plus a mutable layer. `Pipeline::run` is still the biggest function in the crate.

## Recently done

- **Housekeeping sweep** — the theremin is real: `--gesture theremin` runs `ThereminStrategy`, which reads the index tip's x through the same `TipSmoother` EMA and maps it logarithmically onto C2..C6 (`--theremin-low-hz`/`--theremin-high-hz`), with thumb-index spread as amplitude. `GestureEvent::new_signal` came back because something finally produces it. Deleted the never-read `EventType` enum and its field, and `FrameMetrics.frame_id`. `--debug` now prints what is actually sounding (`note C4` or `803 Hz`) instead of `note Some(60)`. AGENTS.md's score threshold, dependency list and scaffolding claims are corrected.
- ~~No rustfmt~~ **done** — the crate is `cargo fmt --check` clean. Worth knowing for the next session: `rustfmt src/main.rs` recurses through every `mod` in the crate, so formatting one file formats the whole tree.
- **MIDI attack discipline** — `src/midi/mod.rs` now holds a `NoteGate` that owns the voice state. The strategy fires an event every frame (30×/s), so the sink used to send `0x90 note vel` each frame and any synth with a percussive envelope machine-gunned a held note. Now: the first event for a note attacks it, further events with the same note only move CC 7 when the volume actually changed, a cell change sends `note-off` on the old note before attacking the new one, and the pipeline's release event (or a pinched, zero-velocity hand) sends exactly one `note-off` — a second release sends nothing, so there is no orphan note-off. `--min-note-ms` (default 120 ms) is the floor: while the held note is younger than that, a new cell is ignored rather than chopping the note into 33 ms slivers, so a hand sliding across the keyboard sounds every fourth key instead of a blur. `--min-note-ms 0` restores attack-on-every-change. Nine tests in `midi/mod.rs` cover the attack, the CC, the floor, the sweep quantisation and the release; the gate returns the raw messages, so it is testable without a synth attached.
- **Feed gate + cadence** — the camera frame is only converted to RGBA when the stage's `feed` toggle is on, and only at `--feed-fps` (default 15) so iced's synchronous texture upload happens half as often. `Pipeline::publish` moves the pixel buffer into the shared snapshot instead of cloning it, and both publish sites (`advance`, `pause`) now share one method.
- **Mirrored capture** — webcam frames are flipped horizontally by default (`--no-mirror` to disable), so the hand and the keyboard agree on left/right.
- **Palm-anchored detector window** — the 192×192 square slides to contain the last palm centre instead of sitting dead-centred; at 640×480 the centred crop only sampled x = 80..560, which is why the outer keys were unreachable. Anchor is EMA-smoothed and dropped after 10 misses.
- **Note keyboard on the stage** — 49 semitone columns with pitch-class colours, a per-note trail with `--fade-ms` decay, a volume bar on the held key, and a play-head line at the index tip.
- **Pitch from the index tip's x** across the frame (previously its height inside the hand box), with a 10 % inset band.
- **Start/Stop button** — pauses the pipeline thread without tearing down camera or sinks, releasing the held note so audio/MIDI go silent.
- **Basic / Advanced tabs** on a scrollable rail; the bottom strip is gone and the stage takes the full middle band.
- **Landmarker crop centred on the true palm** rather than a frame-clamped box, so a clipped hand keeps its fingers in the crop.

## Open gaps worth a look

- `--max-frames` still terminates the pipeline even with the UI open, but Stop/Start has no equivalent headless flag.
- `FrameMetrics.paused` is published but the stage canvas (`Skeleton`) does not use it.
- The golden trace is only reproducible on an otherwise **idle** machine: the detector's ROI anchor is seeded from the previous frame, so load changes the timings, the timings change the anchor, and the anchor changes the detections. Three consecutive `make trace-check` runs matched exactly on a quiet machine; one differed by 36 frames while a `cargo build --release` was competing for cores. Treat it as a regression gate for a quiet machine, not a CI assertion.
- `dashboard/mod.rs`, `dashboard/gauges.rs` and `main.rs` still have no tests.
