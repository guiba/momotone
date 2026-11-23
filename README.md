# Momotone

**Play the air with your hands.** A real-time hand-gesture instrument in Rust: your
hand is the keyboard, the machine makes the sound.

[![CI](https://github.com/guiba/momotone/actions/workflows/ci.yml/badge.svg)](https://github.com/guiba/momotone/actions/workflows/ci.yml)

```
                        ┌─────────────┐
   webcam ─────────────▶│  capture    │  thread 1 · capacity-1 queue, drops stale frames
                        └─────┬───────┘
                              ▼
                        ┌─────────────┐
                        │  inference  │  thread 2 · BlazeHand 192×192 → landmarks 224×224
                        └──┬────┬─────┘   (or one-pass Ultralytics pose: --model yolo)
           hand state ────┘    │
                    │           └──────────────▶ FrameMetrics ─▶ iced dashboard (main thread)
                    ▼
              HandEstimator ─▶ Strategy ─▶ EventBus ─▶ Sinks   thread 3 · cpal voice, MIDI
                    │
                    └──────────────────────────▶ Presenter   thread 4 · ASCII art, preview, dumps
```

- **Vision** — MediaPipe BlazeHand (palm detector → landmark model) or a single-pass
  Ultralytics pose model; both return the same 21 MediaPipe-ordered landmarks.
- **Gesture** — fingertip position on a degree × octave grid picks the note; apparent
  palm size sets loudness. `--gesture theremin` sweeps pitch continuously instead.
- **Sound** — a monophonic cpal voice with attack/release and a pitch glide, or MIDI
  through a note-length gate so a sweep reads as an arpeggio.

## Install

```bash
# macOS — nothing to install beyond Rust; capture and decoding are pure Rust
# (AVFoundation via nokhwa, H.264 via rusty_h264-decoder).

# Ubuntu / Debian — ALSA for cpal's audio output
sudo apt-get install libasound2-dev pkg-config
```

Rust 1.88+ (tested on 1.100). The models are gitignored — put them in `assets/`:

```
assets/hand_detector.onnx
assets/hand_landmarks_detector.onnx
assets/anchors.csv
```

They come from the [Sentis BlazeHand export](https://huggingface.co/unity-sentis/mediapipe-hand-gesture-onnx-float)
([model notes](docs/blazehand-model-notes.md)). For the pose model:

```bash
scripts/get_pose_model.sh fp16      # -> assets/yolo_hand_pose.onnx
```

No camera? `--mode mock` animates a synthetic hand and needs no models at all.

## Quick start

```bash
cargo run                                   # keyboard grid, audio + dashboard
cargo run -- --mode mock --no-ui --ascii    # smoke test: no camera, no models
cargo run -- --gesture theremin             # continuous pitch
cargo run -- --mode midi --port OP-1        # MIDI out
cargo run -- --image hand.png --dump-frame out.png --max-frames 1   # offline detection check
```

Hand on the left of the frame is the low end of the keyboard; closer to the camera is
louder. `Ctrl+C` stops.

## Useful flags

| Flag | Effect |
| :--- | :--- |
| `--gesture position\|theremin` | grid notes vs. continuous pitch |
| `--volume proximity\|pinch\|blend` | what decides loudness (default: proximity) |
| `--scale`, `--root`, `--octaves` | keyboard layout, e.g. `--scale minor pentatonic --root A` |
| `--waveform sine\|triangle\|square\|saw` | synthesiser timbre |
| `--effects lowpass,vibrato,delay` | which stages are in the sound chain (default: `lowpass`; `none` for a bare note) |
| `--model mediapipe\|yolo` | which ONNX backend |
| `--hands one\|two\|probe` | how many hands the tracker looks for. `two` is the default: the anchored crop tracks the hand playing notes and a whole-frame pass finds the second one. `one` is the single-hand instrument; `probe` is the measurement mode below |
| `--camera 1`, `--camera-width 1280` | pick the webcam and capture size |
| `--debug` | DEBUG logging, per-frame timings |
| `--no-ui`, `--ascii`, `--ascii-cols 70` | headless / ASCII renderer |

`--help` lists everything. Logging is `tracing`: `INFO` for what the run is doing, `--debug`
for per-frame detail, and `RUST_LOG` to tune it (`RUST_LOG=info,iced_winit=info` brings back
iced's window/compositor dumps, which are silenced by default).

## Measuring the detector window

The palm detector samples a 192×192 square. Anchoring that square on the palm seen last
frame is great for one hand and blind to a second one, so `--hands probe` runs the anchored
crop *and* a letterboxed whole-frame pass over the same frames and prints, per window:
frames with a hand, mean best score, hands per frame at each NMS separation, frames with
more than one hand, and µs per detector frame.

```bash
make probe          # fixtures/2_hands_clip.mp4, 120 frames
make probe-single   # fixtures/hand_clip.mp4 - the one-hand regression
```

This is a measuring mode, not a playing mode: `probe` runs two detector passes per frame.
`--hands two` is the default for playing.

## Two hands

The tracker reports two hands: the **pitch hand**, which plays notes, and the **modulator
hand**, which shapes the sound. `--hands one` returns to the single-hand instrument.

The pitch hand is found by the palm-anchored crop - the window that magnifies the hand -
and the modulator by a letterboxed whole-frame pass, because a `min(W, H)` square slid to
one hand cannot span two hands that are far apart. Both then run the same landmark model
with their own reuse cache, so the second hand is detected exactly like the first.

With one hand in view the instrument behaves exactly as it always did: that hand plays
pitch *and* drives the whole effect chain. With two, the modulator's proximity, height and
horizontal position drive the chain, and the distance between the palms drives vibrato
depth and echo feedback. Hands keep their identities across frames, and the dashboard's
HANDS card has a **swap hands** button to trade the roles - nothing promotes a hand
automatically, which could hijack the melody mid-sweep.

```bash
make hands-640      # downscale fixtures/2_hands_clip.mp4 to the capture resolution
make hands-trace    # record the two-hand golden trace
make hands-check    # fail if the two-hand run drifts
```

## Tests

`make verify` is the local gate and the CI gate: `cargo check`, `cargo test` (220 tests),
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`.

```bash
make record           # record a 20 s webcam clip into fixtures/hand_clip.mp4 (needs ffmpeg)
make fixture          # 120 frames of that local clip, headless
make trace-check      # fails if the clip's note/volume trace drifts
make trace            # rewrite fixtures/clip_trace.json from the current build
make mock-trace-check # CI's gate: deterministic mock run, no camera, no models
```

The clip and its trace are not in the repo - they are recordings of a real hand. The mock
trace is committed because mock mode is driven by the frame index, so it reproduces on any
machine; that is what CI actually checks.

## Layout

```
src/
  vision/      camera, tracker (MediaPipe), pose model, sampling, landmark reuse
  gesture/     estimator + strategies (keyboard grid, theremin)
  core/        types, grid, scale, geometry, event bus, telemetry, runtime config
  audio/       cpal synthesiser
  midi/        midir sink + note gate
  dashboard/   iced app, canvas layers, gauges
  presentation/ presenter thread, ASCII grid, annotator
```

## Reading

- [CHANGELOG.md](CHANGELOG.md) — what changed, phase by phase
- [AGENTS.md](AGENTS.md) — architecture, data flow, module map, every CLI flag
- [ARCHITECTURE.md](ARCHITECTURE.md) · [MASTER_PLAN.md](MASTER_PLAN.md) · [ROADMAP.md](ROADMAP.md)
- [bench/baseline.md](bench/baseline.md) — measured stage timings
