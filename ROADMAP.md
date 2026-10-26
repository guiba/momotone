# Roadmap — gesture_rust

Short list of what to do next. Effort tags: **S** (a few hours), **M** (a day-ish), **L** (multi-day).
Ordered roughly by value-to-effort. `AGENTS.md` has the "Known Gaps" list this roadmap is built on.

## Now — bugs blocking usable output
- ~~Fix the MIDI range: strategy emits notes 36–84, `MidiOutputEngine` adds +60 → up to 144 (> 127)~~ **done** — the +60 offset is gone; notes are used as-is (36–84, C2–C6) and clamped to 0–127
- ~~Send note-off events: notes are re-triggered every frame with no release~~ **done** — the pipeline tracks `gesture_active` and dispatches a `no_gesture()` release when the gesture ends; `MidiOutputEngine` now holds `held_note` and sends `0x80` before a new note and on release, and the synthesizer clears its voice. Detector gate raised to 0.7 (`--score-threshold`)
- Add hysteresis / quantisation to the pitch mapping so a shaky hand does not jitter across semitones — **S**

## Performance
- ~~Replace hand-written bilinear `sample_affine_to_tensor()` (tracker.rs) with OpenCV `warpAffine`/`resize` + one bulk BGR→RGB→f32 conversion~~ **done** — `resize` on the crop (warpAffine fallback when the crop runs off the frame), one bulk BGR→RGB `/255` pass, and the sampler now returns the mapping it actually applied (`Resampled { tensor, map }`) so box/landmark conversion stays consistent. Measured with `cargo test --release -- --include-ignored bench_sampling`: 192×192 190 µs → 109 µs, 224×224 270 µs → 66 µs per stage
- ~~Reuse one input buffer/tensor per stage instead of allocating + copying (`vec![0.0f32; …]`, `Array4::from_shape_vec`, `Value::from_array`) every frame~~ **done** — `StageInput` (OpenCV scratch Mat + f32 buffer) is allocated once in `HandTrackerImpl::new` and refilled in place; ort now gets a zero-copy `TensorRef` view of it and `ndarray` was dropped from `Cargo.toml`. Honest note: `cargo test --release -- --include-ignored bench_sampling` shows only ~3% (192×192 101.5 µs reused vs 104.6 µs fresh; 224×224 59.5 µs vs 58.8 µs) because malloc reuses the freed blocks and ort was already wrapping the `Vec` without copying — the real per-frame cost is now `cv::resize` itself, so the capture-resolution item below is the bigger win
- ~~Set capture resolution/FPS explicitly in `WebcamSource::new` and feed both stages a downscaled frame (e.g. 640×480) instead of the raw camera frame~~ **done** — `WebcamSource::new(camera_id, width, height)` requests 640×480 by default (`--camera-width` / `--camera-height`); FPS is still not pinned
- Skip the landmarker when the detector box is unchanged / below a stability threshold, or run it every Nth frame and reuse landmarks — **M**
- Remove per-frame allocations in `process_frame`: `outputs.into_iter().collect()` and `landmarks.clone()` — **S**
- ~~Get `imshow` + `wait_key` (and `--dump-frame`, which writes a PNG *every* iteration) out of the hot loop; add `--no-display`~~ **done** — the OpenCV window is gone (only `--image` dumps use it) and `--no-ui` runs the pipeline headless
- ~~Split the pipeline into stages (capture → inference → sinks → display) on threads; either actually use `tokio` (declared in `Cargo.toml` but unused) or delete the dependency~~ **partly done** — the camera/inference/sink loop lives in `src/pipeline.rs` on its own thread, and `tokio` is now iced's executor; capture, inference and sinks still share one thread
- Audio callback: drop the per-block `Mutex` lock (atomics or lock-free shared state), ramp frequency to avoid zipper clicks, add ADSR + note-off — **M**
- Add a release profile (`lto = true`, `target-cpu = native`) and an `--fps`/per-stage latency printout so improvements are measurable — **S**
- Clear the ~14 dead-code warnings so the real hot path is obvious: `sources/mock.rs`, `SimplePinchStrategy`, `FrequencyStrategy`, `Source`/`Tracker` traits, `no_gesture()` — **S**

## Features
### Gesture layer
- Landmark smoothing (EMA / simple Kalman-lite) before any strategy runs — **S**
- Discrete gesture set: open hand, fist, pinch, peace; per-gesture mapping configurable instead of hardcoded — **M**
- Finger-extension detection (tip vs. MCP distance) so each finger is an on/off switch — **M**
- Gesture "chords": combine several gestures into a chord/scale/scale-mode selection — **M**

### Output sinks
- OSC sink (implement `Sink`, push into `sinks` in `main.rs`) — **S**
- MIDI polyphony with a voice pool and proper note release — **M**
- JSON/CSV event log for offline analysis of gesture traces — **S**

### Input & evaluation
- `--video <file>` playback source so runs are reproducible — **M**
- Benchmark harness: recorded clip + FPS/accuracy report, so perf work is verifiable — **M**
- `--calibrate` mode that samples the user's hand range and stores per-user min/max — **M**

### Visualization & UX
- ~~Draw the hand box (computed but never rendered), HUD FPS + detector confidence, gesture name~~ **done** — the iced dashboard (`src/dashboard/`) replaced OpenCV `imshow`: skeleton canvas, box toggle, note/volume/confidence meters, per-stage latency, FPS, jitter, finger extension, rolling sparklines and an event log
- ~~Split the pipeline into stages (capture → inference → sinks → display) on threads~~ **partly done** — the frame loop now runs on its own thread (`src/pipeline.rs`) and the iced UI owns the main thread; capture/inference/sinks are still one thread
- ~~Feed the camera frames into the dashboard as an `image` widget (`image::Handle::Rgba` from the `Mat` pixels) instead of a separate OpenCV window~~ **done** — the stage canvas draws the RGBA frame and the skeleton in the same iced window
- Landmark smoothing (EMA) before drawing and before strategies, with the smoothing strength shown in the rail — **S**
- Note visualisation: a mini keyboard / octave strip showing the current note and recent notes, plus MIDI note-on/off state and active voice count — **M**
- Audio visualiser: render the actual oscillator envelope or an RMS trace from the cpal callback — **M**
- Finger-extension chips that already exist in the rail → turn them into discrete gesture triggers (open hand, fist, pinch) — **M**
- Remove the remaining dead code so the dashboard's data path is obvious: `sources/mock.rs`, `Visualizer::draw_hud`, `SimplePinchStrategy`, `FrequencyStrategy`, `no_gesture()`, `Source`/`Tracker` traits — **S**

### Tooling
- `cargo fmt` + `clippy` + `cargo check` in CI — **S**
- Tracker fixture tests using `--image` snapshots (gold box + 21 landmarks) — **M**
- Replace `println!`/`eprintln!` with `tracing` — **S**
- TOML config file for model paths, thresholds, note ranges (CLI overrides it) — **M**

## Stretch / later
- GPU or other ONNX Runtime execution provider for `ort` — **L**
- Single-shot hand-mesh model instead of the two-stage detector + landmarker — **L**
- Two hands: NMS over top-k anchors instead of a single argmax, `Vec<HandDetectionResult>`, left/right roles (pitch vs. modulation) — **L**
- Retrain/re-export a lighter BlazeHand variant for lower latency — **L**
- Use `Landmark.z` for depth-based gestures (currently read nowhere) — **M**
