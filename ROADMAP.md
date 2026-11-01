# Roadmap — gesture_rust

Effort tags: **S** (a few hours), **M** (a day-ish), **L** (multi-day). Ordered by value-to-effort.
Status: measured on `master` after the mirror/ROI/UI/feed work (see "Recently done" at the bottom).

## Measured baseline

| What | Number | Where |
| :--- | :--- | :--- |
| Frame loop (MediaPipe, live camera) | **30.6 fps**, 33.2 ms/frame — capture 32.3 ms, inference 8.0 ms (detector 8.0 ms, landmarker 0 µs when the reuse cache is serving), 0 dropped | `cargo run --release -- --no-ui --max-frames 200 --debug` |
| Frame loop (pose model) | **21 fps**, 41 ms/frame — capture 29 ms, inference 69 ms, 13 of 73 dropped | same with `--model yolo --score-threshold 0.3` |
| Camera read | ~31 ms; the loop is paced by the driver, not by inference | `--debug` end-of-run summary |
| Detector stage (192×192, hand absent) | 9.5 ms | same run, `--debug` |
| Detector stage (192×192, fixture image) | 7.2 ms | `cargo run --release -- --image output_frame.png --debug --max-frames 40` |
| Detector input sampling (192×192) | 133 µs (reused buffer) vs 247 µs legacy | `cargo test --release -- --include-ignored bench_sampling --nocapture` |
| Landmarker input sampling (224×224) | 94 µs vs 367 µs legacy | same bench |
| Frame snapshot copy | 0.44 µs | not worth optimising |
| RGBA conversion (`frame_to_rgba`) | 0.30 ms/frame — only when the feed is on *and* due | `cargo test --release -- --include-ignored bench_feed_upload --nocapture` |
| Snapshot publish carrying a full frame | 0.020 ms cloned; 0 once the pixel buffer is moved instead of cloned | `cargo test --release -- --include-ignored bench_publish_snapshot --nocapture` |
| Mirroring | **0.018 ms/frame**, no allocation after the first frame | `cargo test --release -- --include-ignored bench_mirror_flip` |
| Mock-mode clock | **40.8 ms/frame measured** against the 33 ms requested | `--mode mock --no-ui --max-frames 120 --debug` |
| rustc warnings | 0 (only the third-party `block v0.1.6` future-incompat notice) | `cargo build` |
| clippy lints | 1 — `chunks_exact_to_as_chunks`, `src/pipeline.rs:757` | `cargo clippy` |
| Dead code | 1 flag, `Visualizer::draw_skeleton`, which *is* used by `--image` — a call-graph false positive | `aft_inspect` |

## Codebase census (this sweep)

- **24 source files, ~6.9k lines.** `pipeline.rs` (1383) and `vision/tracker.rs` (790) are the two big ones; `dashboard/mod.rs` is 663.
- **91 tests pass, 4 benches are `#[ignore]`d.** Coverage is concentrated in `pipeline.rs` (21 tests), `strategies` (14 across the keyboard and the theremin), `tracker` (7), `yolo_pose` (5), `midi` (9), `grid`/`scale`/`instrument`/`synthesizer` (5 each).
- **Zero tests** in `dashboard/mod.rs` (663 lines), `dashboard/skeleton.rs` (514), `dashboard/gauges.rs` (139), `main.rs` (357), `visualizer/mod.rs` (101), `core/event_bus.rs`. That is ~1.5k lines of UI and plumbing code with no test at all.
- **Cyclomatic complexity hotspots:** `Skeleton::draw` 20, `Pipeline::advance` 17, `Pipeline::run` 17, `Dashboard::update` 16, `keyboard()` 16, `sample_affine_to_tensor` 16, `run_detector` 12, `root_class` 11, `main` 10, `decode_pose` 10.
- **Duplicate groups:** three ~20-line pairs in `vision/tracker.rs` — all inside `bench_sampling`, i.e. benchmark scaffolding, not production duplication.
- **Import cycles:** 0.

## New findings worth scheduling

1. **Mock mode runs at 25 fps, not 30** — `tick()` sleeps a flat `MOCK_CADENCE_MS`, and macOS `thread::sleep(33 ms)` actually returns after ~40.8 ms. Every mock telemetry number (fps, latency, trail decay) is therefore 23 % off. Fix with a deadline clock: compute `next = start + 33 ms`, sleep only the remainder, and drop the sleep when it is already past. **S**
2. ~~The MIDI sink re-sends `note-on` every frame for a held note~~ **done** — see "Recently done": the sink now carries a `NoteGate`, so a held note attacks once, volume travels as CC 7, and `--min-note-ms` quantises a fast sweep into an arpeggio.
3. **The oscillator is not band-limited and frequency changes are not phase-continuous** — `Waveform::Square`/`Saw` are naive `if phase < 0.5` / `2*phase - 1` shapes, which alias badly above a few hundred Hz, and setting `voice.frequency` mid-note jumps the phase, producing clicks when the hand sweeps the keyboard. Proposal: BLIT/synthesised square, and glide the frequency over ~10-30 ms (portamento) instead of snapping it. **M**
4. **`ort` sessions are created with default settings** — no `intra_op_thread_count`, no provider selection. With capture, inference, sinks and the UI all on their own threads, ORT oversubscribing the cores is a plausible reason the detector alone costs 8 ms of the 33 ms budget. Measure `--debug` timings with the thread count pinned to 1, 2 and 4. **S**
5. ~~Vestigial payload: nobody emits `GesturePayload::Signal`~~ **done** — it is now real: `--gesture theremin` (`src/strategies/theremin.rs`) maps the index tip's x to a logarithmic C2..C6 band and the thumb-index spread to amplitude. `EventType`/`GestureEvent.event_type`, which nothing ever read, were deleted.
6. ~~Dead fields in the shared snapshot~~ **done** — `FrameMetrics.frame_id` and `GestureEvent.event_type` are gone. `scale::is_in_scale` stays, deliberately: it is the test-side invariant that every cell `GridLayout` builds really belongs to the scale it was given.
7. ~~Doc drift~~ **done** — `AGENTS.md` now says `DEFAULT_SCORE_THRESHOLD = 0.6` (the code's value), drops the phantom `ndarray 0.17` dependency, describes the deleted scaffolding as deleted, and lists `--gesture` / `--theremin-low-hz` / `--theremin-high-hz` / `--min-note-ms`.
8. **No detector hysteresis** — the argmax anchor crosses the threshold from frame to frame (measured: 0.401 then 0.463 on the same empty scene with `--score-threshold 0.6`), so detection flickers near the gate. Require N consecutive detections before a gesture starts and a short grace window before the release event. **S**
9. **Camera resolution is larger than the model ever sees** — capture is 640×480 but the detector samples a 192×192 square and the landmarker a 224×224 one. Try `--camera-width 320 --camera-height 240` and compare `capture` in the `--debug` summary; the camera read (32 ms) *is* the loop's pacing, so cheaper pixels are the only remaining lever on end-to-end latency. **S**
10. **The UI's only unshared state is the trail** — `note_trail`, `volume_history` and `confidence_history` live in `Dashboard` while everything else round-trips through `FrameMetrics`. Making the trail part of the published snapshot would let the pipeline own the decay and would make the keyboard drawable from a pure function (testable). **S**
11. **No CI, no clippy gate** — ~~no rustfmt~~ **done**: the whole crate is now `cargo fmt --check` clean (the tree predated formatting and got swept as a side effect of the housekeeping commit). Clippy still has one live lint (`chunks_exact_to_as_chunks`, `src/pipeline.rs:757`). A CI job is the cheapest way to keep the warning count at zero. **S**
12. **Latency is reported as an EMA only** — `fps`, `frame_ms`, `capture_ms` and `latency_ms` are last-frame values; a dropped-frame spike is invisible except in the counter. A rolling p50/p95 of each stage would make regressions obvious from the dashboard. **S**

## Now — highest value next

1. ~~MIDI note-on/off discipline~~ (finding 2) + **retrigger on cell change** — one attack per cell transition, with a `--min-note-ms` floor so a fast sweep becomes an arpeggio rather than a blur. **done for MIDI; the audio sink still glides** (see finding 3).
2. **Mock-mode deadline clock** (finding 1) — mock is the only way to exercise the UI without a webcam, so its pacing has to be honest. **S**
3. **Tracker fixture tests** using `--image` snapshots (gold box + 21 landmarks) so the ONNX stages are tested, not just the geometry helpers. **M**
4. **CI**: `cargo fmt --check` + `clippy` + `cargo check` (finding 11). **S**
5. **Audio: band-limited waveforms + ADSR + portamento** (finding 3) — the biggest jump from "theremin" to "instrument". **M**
6. **Detector hysteresis** (finding 8). **S**
7. **ORT thread pinning** (finding 4). **S**
8. **Cache the keyboard backdrop** — `keyboard()` re-fills all 49 base cells plus separators every UI tick; the static part (cells, separators, octave labels) could be drawn once or only when the panel size changes. **S**
9. ~~Housekeeping sweep~~ (findings 5, 6, 7) **done** — see "Recently done".
10. ~~Kill the per-frame clones~~ **done, but it was not the bottleneck** — the tracker now moves the landmark Vec into the result instead of cloning it, and `Pipeline::tick` refills `metrics.landmarks` / `previous_landmarks` with `clone_from` so their buffers are reused. Measured: copying the whole `FrameMetrics` snapshot costs **0.44 µs/frame** (0.0015 % of a 33 ms frame), so a hand-written `copy_from` was not worth the maintenance risk of a field-by-field copy drifting out of sync with the struct.
11. ~~Pitch hysteresis / smoothing~~ **done** — the index-tip x runs through an EMA (`PITCH_EMA_ALPHA = 0.35`) before quantisation, and the smoother resets when the hand is lost. Test `the_smoothing_kills_jitter_but_follows_a_sweep` compares note flapping with and without it.
12. ~~Release profile + per-stage latency printout~~ **done** — `[profile.release] lto = "fat"`, `codegen-units = 1`, and `--debug` now prints `frame 31.07 ms (detector 9548 us, landmarker 0 us)`.
13. ~~Clear the 13 dead-code warnings~~ **done** — deleted `sources/mock.rs` (`MockSource`, `MockSourceImpl`, `SourceTrait`), `SimplePinchStrategy`, `FrequencyStrategy`, `Visualizer::draw_hud`, the core `Source`/`Tracker` traits, `GestureEvent::new_signal` and `MidiOutputEngine::control_change`; `StreamWrapper`'s unread field is now `#[allow(dead_code)]` with a comment (holding the stream *is* the point). `cargo build` is warning-free except the third-party `block v0.1.6` future-incompat notice.

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
- **Audio callback lock** — the cpal callback takes a `Mutex<SynthesizerState>` per audio block (a callback every few milliseconds, i.e. hundreds of lock/unlock pairs per second). Move to atomics or a double-buffered state so the hot path is lock-free. **M**
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
- **Waveform + ADSR in `Voice`** — sine/triangle/square/saw are in; attack/decay, band-limiting and detuned pairs are still open. **M**
- **Vertical band calibration** — learn the y range the user actually occupies from recent landmark bounding boxes, so the top row is not unreachable. **M**
- **Named instrument presets** — bundle scale + root + octave span + waveform + envelope (e.g. *Kalimba*: C pentatonic, pluck). **S**
- **Velocity curve** — volume is currently a linear read of thumb↔index spread saturating at 60 % of the hand box. An exponent or a per-user curve (see `--calibrate`) would make the instrument feel less like a gate. **S**

### Gesture layer
- **Landmark smoothing (EMA / Kalman-lite)** before any strategy runs — the tip is smoothed, the other 20 joints are not, so `spread`, `extension` and `tilt` are noisier than the pitch. **S**
- **Detection hysteresis** (finding 8) — debounce the confidence gate instead of a per-frame argmax compare. **S**
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
- **`--video <file>` playback source** so runs are reproducible. **M**
- **Benchmark harness**: recorded clip + FPS/accuracy report. **M**
- **`--calibrate`** — sample the user's comfortable hand range and store per-user min/max, replacing the fixed `NOTE_MARGIN`. **M**
- **Stage latency percentiles** (finding 12) — p50/p95 per stage in the Performance card. **S**

### Visualisation & UX
- **Keyboard shortcuts** for Start/Stop and the toggles (iced key events) so the app is operable without hunting for buttons. **S**
- **Stage overlay while paused** — a "stopped" banner on the keyboard so the state is obvious from a glance, not just in the Status card. `Skeleton` already receives `lost_ms` but not `paused`. **S**
- **In-UI fade control** — expose `--fade-ms` as a slider/stepper in the Advanced tab. **S**
- **Show `--feed-fps` in the header** — the preview cadence is currently invisible, so a 15 fps feed reads as a bug rather than a setting. **S**
- **Note-name labels on the keyboard** (currently only the octave Cs); highlight the current pitch class. **S**
- **Audio visualiser** — render the oscillator envelope or an RMS trace from the cpal callback. **M**
- **Voice count / MIDI note-on-off state** in the rail. **S**

### Tooling
- **CI**: `cargo fmt --check` + `cargo clippy` + `cargo check` (finding 11). **S**
- **Tracker fixture tests** using `--image` snapshots (gold box + 21 landmarks) so the inference path is tested, not just the geometry helpers. **M**
- **UI unit tests** — `Mapping`/letterbox math, `cell_index`, `fade_trail` and the note-name tables are pure functions with no test today (finding in the census). **S**
- **Replace `println!`/`eprintln!` with `tracing`**. **S**
- **TOML config** for model paths, thresholds, note ranges, with CLI overrides. **M**
- **Split the complexity hotspots** — `Skeleton::draw` (20) and `Pipeline::advance` (17) are the two functions where a change touches five unrelated concerns; extracting `draw_backdrop`/`draw_skeleton`/`draw_labels` and `update_metrics`/`dispatch_gesture` would make each testable in isolation. **M**

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
- Tests cover strategies, geometry, sampling, reuse, the trail, the capture stage, the release path and the MIDI note gate; nothing exercises the ROI window logic beyond `detector_crop`, no test runs the ONNX stages on a fixture image, and the iced dashboard and the visualizer have none at all.
