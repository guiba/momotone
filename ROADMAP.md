# Roadmap — gesture_rust

Effort tags: **S** (a few hours), **M** (a day-ish), **L** (multi-day). Ordered by value-to-effort.
Status: measured on `master` after the mirror/ROI/UI work (see "Recently done" at the bottom).

## Measured baseline

| What | Number | Where |
| :--- | :--- | :--- |
| Frame loop | **30 fps** (camera-limited) | `cargo run --release -- --debug --no-ui` |
| Detector input sampling (192×192) | 133 µs (reused buffer) vs 247 µs legacy | `cargo test --release -- --include-ignored bench_sampling --nocapture` |
| Landmarker input sampling (224×224) | 94 µs vs 367 µs legacy | same bench |
| RGBA upload when the feed is off | ~1.2 MB/frame built then discarded | `Pipeline::tick` → `frame_to_rgba` |
| Mirroring | one extra `Mat` allocation per frame (~920 KB) | `WebcamSource::get_frame` |
| Dead-code warnings | 13 | `cargo build` |

Sampling is **not** the bottleneck: ~227 µs per frame out of a ~33 ms budget. Inference (`ort` sessions) and per-frame copying are.

## Now — highest value next

1. **Gate the RGBA upload on the feed toggle** — `frame_to_rgba` runs every frame whenever the UI is open, and the dashboard `clear()`s the pixels when `show_feed` is false. **S**
2. ~~Kill the per-frame clones~~ **done, but it was not the bottleneck** — the tracker now moves the landmark Vec into the result instead of cloning it, and `Pipeline::tick` refills `metrics.landmarks` / `previous_landmarks` with `clone_from` so their buffers are reused. Measured: copying the whole `FrameMetrics` snapshot costs **0.44 µs/frame** (0.0015 % of a 33 ms frame), so a hand-written `copy_from` was not worth the maintenance risk of a field-by-field copy drifting out of sync with the struct.
3. ~~Pitch hysteresis / smoothing~~ **done** — the index-tip x runs through an EMA (`PITCH_EMA_ALPHA = 0.35`) before quantisation, and the smoother resets when the hand is lost. Test `the_smoothing_kills_jitter_but_follows_a_sweep` compares note flapping with and without it.
4. **Release profile + per-stage latency printout** — no `lto`/`target-cpu` in `Cargo.toml`, and `--debug` prints fps but not `detector_us`/`landmarker_us`. Add both so perf work is measurable. **S**
5. **Clear the 13 dead-code warnings** — `sources/mock.rs` (`MockSource`, `MockSourceImpl`, `SourceTrait`), `SimplePinchStrategy`, `FrequencyStrategy`, `Visualizer::draw_hud`, core `Source`/`Tracker` traits, `GestureEvent::new_signal`, `MidiOutputEngine::control_change`, `StreamWrapper`'s unread field. **S**

## Performance

- **Skip the landmarker when the hand is static** — if the detector box and anchor are unchanged (or confidence is within a stability band), reuse the previous landmarks instead of running the 224×224 model every frame. **M**
- **Mirror flip in place** — `core::flip(&frame, &mut frame, 1)` avoids the second `Mat`; alternatively flip the landmarks' x in the tracker and skip the pixel copy altogether. **S**
- **Audio callback lock** — the cpal callback takes a `Mutex<SynthesizerState>` per audio block. Move to atomics / a double-buffered state so the hot path is lock-free. **M**
- **Split capture, inference and sinks onto their own threads** — one thread still does camera read → detector → landmarker → strategy → sinks. **M**
- **Trim `tokio`** — declared with `features = ["full"]` and only used as iced's async executor; a slimmer feature set removes an unnecessary dependency weight. **S**
- **Cache the keyboard backdrop** — `keyboard()` re-fills all 49 base cells plus separators every UI tick; the static part (cells, separators, octave labels) could be drawn once or only when the panel size changes. **S**
- **GPU / alternative ONNX Runtime execution provider** for `ort`. **L**
- **Single-shot hand model** instead of detector + landmarker, or a retrained lighter BlazeHand variant. **L**
- **Pin capture FPS** — resolution is requested (`--camera-width`/`--camera-height`) but FPS is not; `CAP_PROP_FPS` is still unset. **S**

## Features

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
