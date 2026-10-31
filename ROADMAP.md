# Roadmap — gesture_rust

Effort tags: **S** (a few hours), **M** (a day-ish), **L** (multi-day). Ordered by value-to-effort.
Status: measured on `master` after the mirror/ROI/UI work (see "Recently done" at the bottom).

## Measured baseline

| What | Number | Where |
| :--- | :--- | :--- |
| Frame loop | **30 fps**, 31 ms/frame | `cargo run --release -- --debug --no-ui` |
| Camera read | ~21 ms of that frame | `frame_ms - detector_us` |
| Detector stage (192×192, hand absent) | **9.5 ms** | same run, `--debug` |
| Detector stage (192×192, fixture image) | 7.2 ms | `cargo run --release -- --image output_frame.png --debug --max-frames 40` |
| Detector input sampling (192×192) | 133 µs (reused buffer) vs 247 µs legacy | `cargo test --release -- --include-ignored bench_sampling --nocapture` |
| Landmarker input sampling (224×224) | 94 µs vs 367 µs legacy | same bench |
| Frame snapshot copy | 0.44 µs | not worth optimising |
| RGBA upload when the feed is off | 0.16 ms/frame built then discarded (now gated) | `cargo test --release -- --include-ignored bench_feed_upload --nocapture` |
| Mirroring | **0.018 ms/frame**, no allocation after the first frame | `cargo test --release -- --include-ignored bench_mirror_flip` |
| Dead-code warnings | 13 | `cargo build` |

Inference is the bottleneck: the detector alone is ~10 ms and the landmarker runs every
frame a hand is present. Sampling is ~0.25 ms out of a ~31 ms frame budget.

## Now — highest value next

1. **Gate the RGBA upload on the feed toggle** — `frame_to_rgba` runs every frame whenever the UI is open, and the dashboard `clear()`s the pixels when `show_feed` is false. **S**
2. ~~Kill the per-frame clones~~ **done, but it was not the bottleneck** — the tracker now moves the landmark Vec into the result instead of cloning it, and `Pipeline::tick` refills `metrics.landmarks` / `previous_landmarks` with `clone_from` so their buffers are reused. Measured: copying the whole `FrameMetrics` snapshot costs **0.44 µs/frame** (0.0015 % of a 33 ms frame), so a hand-written `copy_from` was not worth the maintenance risk of a field-by-field copy drifting out of sync with the struct.
3. ~~Pitch hysteresis / smoothing~~ **done** — the index-tip x runs through an EMA (`PITCH_EMA_ALPHA = 0.35`) before quantisation, and the smoother resets when the hand is lost. Test `the_smoothing_kills_jitter_but_follows_a_sweep` compares note flapping with and without it.
4. ~~Release profile + per-stage latency printout~~ **done** — `[profile.release] lto = "fat"`, `codegen-units = 1`, and `--debug` now prints `frame 31.07 ms (detector 9548 us, landmarker 0 us)`.
5. ~~Clear the 13 dead-code warnings~~ **done** — deleted `sources/mock.rs` (`MockSource`, `MockSourceImpl`, `SourceTrait`), `SimplePinchStrategy`, `FrequencyStrategy`, `Visualizer::draw_hud`, the core `Source`/`Tracker` traits, `GestureEvent::new_signal` and `MidiOutputEngine::control_change`; `StreamWrapper`'s unread field is now `#[allow(dead_code)]` with a comment (holding the stream *is* the point). `cargo build` is warning-free except the third-party `block v0.1.6` future-incompat notice.

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

- ~~Skip the landmarker when the hand is static~~ **done** — `LandmarkCache` reuses the previous keypoints while the palm is within 3 px and 3 % of its last size, capped at 3 consecutive frames so a hand that reshapes without moving its palm is never served stale landmarks for long, and cleared whenever the detector loses the hand. The landmarker stage costs ~4-5.4 ms when it runs, so a still hand saves most of it. Verify with `cargo run -- --debug` (`palm steady, reusing landmarks`) and compare against `--landmarker-always`. **Note:** fps is camera-paced at ~30, so the win is latency and CPU headroom, not frame rate.
- ~~Mirror flip in place~~ **done, but it was never a bottleneck** — the bindings take two separate borrows, so `flip(frame, frame, 1)` is not expressible; `WebcamSource` now carries a scratch `Mat` and swaps it with the captured frame, so the destination allocation happens once instead of every frame. Measured 0.0176 ms/frame (fresh Mat) vs 0.0174 ms/frame (reused scratch): the allocator recycles the 920 KB block, so the saving is ~0.0002 ms/frame. Flipping the landmarks' x instead of the pixels would skip the memcpy too, but that is at most 0.018 ms of a 31 ms frame — inference and the camera read are where the time is. Verified by `mirroring_reverses_the_columns` and `a_second_flip_reuses_the_scratch_buffer_instead_of_allocating`.
- **Audio callback lock** — the cpal callback takes a `Mutex<SynthesizerState>` per audio block. Move to atomics / a double-buffered state so the hot path is lock-free. **M**
- **Split capture, inference and sinks onto their own threads** — one thread still does camera read → detector → landmarker → strategy → sinks. **M**
- **Trim `tokio`** — declared with `features = ["full"]` and only used as iced's async executor; a slimmer feature set removes an unnecessary dependency weight. **S**
- **Cache the keyboard backdrop** — `keyboard()` re-fills all 49 base cells plus separators every UI tick; the static part (cells, separators, octave labels) could be drawn once or only when the panel size changes. **S**
- **GPU / alternative ONNX Runtime execution provider** for `ort`. **L**- **Single-shot hand model** instead of detector + landmarker, or a retrained lighter BlazeHand variant. **L**
- ~~Pin capture FPS~~ **done** — `WebcamSource::new` sets `CAP_PROP_FPS` (`--camera-fps`, default 30) and `--debug` prints what was requested versus what the driver reports, so a camera defaulting to 60 fps no longer doubles the inference load unnoticed.

## Features

### Instruments & layouts
- ~~Quantise pitch to a scale~~ **done** — `src/core/scale.rs` holds seven presets (major/minor pentatonic, major, minor, dorian, whole tone, chromatic) plus root-name parsing; `--scale` / `--root` choose it. Every reachable cell is consonant with every other one.
- ~~Degree × octave matrix~~ **done** — `src/core/grid.rs` builds a `GridLayout` (columns = scale degree, rows = octave, top row highest) centred inside C2–C6 and shrunk if the requested span cannot fit. `FingerPositionStrategy` EMA-smooths the index tip in **both** axes and maps it to a cell; the canvas draws the same cells with the same `NOTE_MARGIN` inset, labels octave rows, and names each cell when it is big enough. `--octaves` sets the row count (default 3).
- **Retrigger on cell change** — one attack per cell transition instead of a continuous re-tune, with a `--min-note-ms` floor so a fast sweep becomes an arpeggio rather than a blur. **S**
- **Waveform + ADSR in `Voice`** — sine/triangle/square/saw plus attack/decay, so a grid press reads as an instrument instead of a theremin. Detuned pairs give a pad. **M**
- ~~Dashboard pickers~~ **done** — an `INSTRUMENT` card at the top of the basic rail with pick lists for scale, root, octave span and waveform. They write into one `Arc<Mutex<InstrumentConfig>>` (`src/core/instrument.rs`); the pipeline compares its `version` at the top of every tick and, only when it moved, re-keys the strategy and pushes the new timbre to the sinks. No restart needed.
- ~~Waveform~~ **partly done** — `Voice` now supports sine, triangle, square and saw (`Waveform::sample`), applied to the held voice as soon as the picker changes. Attack/decay envelope and detuned pairs are still open. **M**
- **Vertical band calibration** — learn the y range the user actually occupies from recent landmark bounding boxes, so the top row is not unreachable. **M**
- **Named instrument presets** — bundle scale + root + octave span + waveform + envelope (e.g. *Kalimba*: C pentatonic, pluck). **S**

### Gesture layer
- **Landmark smoothing (EMA / Kalman-lite)** before any strategy runs — also fixes the jitter in item 3 above. **S**
- **Discrete gesture set**: open hand, fist, pinch, peace, with per-gesture mapping configured rather than hardcoded. **M**
- **Finger extension as switches** — `hand_geometry()` already computes per-finger extension; turn each into an on/off gate. **M**
- **Gesture chords** — combine several gestures into chord/scale selection. **M**
- **Use `Landmark.z`** — depth is parsed but read by nothing. **M**
- **Two hands** — NMS over top-k anchors instead of a single argmax, `Vec<HandDetectionResult>`, left/right roles (pitch vs modulation). **L**

### Output sinks
- **OSC sink** — implement `Sink` and push it into `sinks` in `main.rs`. **S**
- **MIDI polyphony** — `MidiOutputEngine` is monophonic (`held_note: Option<u8>`); add a voice pool. **M**
- **JSON/CSV event log** for offline analysis of gesture traces. **S**

### Input & evaluation
- **`--video <file>` playback source** so runs are reproducible. **M**
- **Benchmark harness**: recorded clip + FPS/accuracy report. **M**
- **`--calibrate`** — sample the user's comfortable hand range and store per-user min/max, replacing the fixed `NOTE_MARGIN`. **M**

### Visualisation & UX
- **Keyboard shortcuts** for Start/Stop and the toggles (iced key events) so the app is operable without hunting for buttons. **S**
- **Stage overlay while paused** — a "stopped" banner on the keyboard so the state is obvious from a glance, not just in the Status card. **S**
- **In-UI fade control** — expose `--fade-ms` as a slider/stepper in the Advanced tab. **S**
- **Note-name labels on the keyboard** (currently only the octave Cs); highlight the current pitch class. **S**
- **Audio visualiser** — render the oscillator envelope or an RMS trace from the cpal callback. **M**
- **Voice count / MIDI note-on-off state** in the rail. **S**

### Tooling
- **CI**: `cargo fmt` + `clippy` + `cargo check`. **S**
- **Tracker fixture tests** using `--image` snapshots (gold box + 21 landmarks) so the inference path is tested, not just the geometry helpers. **M**
- **Replace `println!`/`eprintln!` with `tracing`**. **S**
- **TOML config** for model paths, thresholds, note ranges, with CLI overrides. **M**

## Recently done

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
- Tests cover strategies, geometry, sampling, the trail and the release path; nothing exercises the ROI window logic beyond `detector_crop`, and no test runs the ONNX stages on a fixture image.
