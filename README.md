# Momotone

**Play the air with your hands.** A real-time hand-gesture instrument in Rust: your
hand is the keyboard, the machine makes the sound.

```
webcam ─▶ OpenCV ─▶ ONNX (BlazeHand detector + landmark detector, via ort)
        ─▶ HandEstimator ─▶ Strategy ─▶ EventBus ─▶ cpal synthesiser / MIDI
                                  └────────────▶ iced dashboard (HUD, keyboard, ASCII)
```

- **Vision** — two-stage MediaPipe BlazeHand (192×192 palm detector → 224×224 landmark
  model) or a single-pass Ultralytics pose model (`--model yolo`), both returning the
  same 21 MediaPipe-ordered landmarks.
- **Gesture** — the fingertip's position on a degree × octave grid selects the note;
  apparent palm size (`proximity`, the default) decides loudness. `--gesture theremin`
  sweeps pitch continuously instead.
- **Output** — a monophonic cpal voice with attack/release/pitch glide, or MIDI through
  a note-length gate so a sweep across the keyboard reads as an arpeggio.

## Requirements

- Rust 1.88+ (tested on 1.100), Homebrew LLVM (the OpenCV bindings are generated with clang)
- OpenCV 4.x installed
- A webcam (or `--video <clip>` / `--mode mock` for a synthetic hand with no camera)
- `assets/hand_detector.onnx`, `assets/hand_landmarks_detector.onnx`, `assets/anchors.csv`
  — gitignored, fetched locally. `scripts/get_pose_model.sh fp16` downloads a pose model.

## Run

```bash
cargo run                                        # keyboard grid, audio output + dashboard
cargo run -- --gesture theremin                  # continuous pitch
cargo run -- --mode midi --port OP-1             # MIDI instead of audio
cargo run -- --mode mock --no-ui --ascii        # no webcam, ASCII art, headless
cargo run -- --image photo.png --dump-frame out.png --max-frames 1   # offline detection check
```

## Verify

`make verify` is the gate: `cargo check` + tests + `cargo fmt --check` +
`cargo clippy --all-targets -- -D warnings`. There is no CI; everything runs locally.

```bash
make fixture      # 300 frames of fixtures/hand_clip.mp4, headless
make trace-check  # fails if the note/volume trace drifts from the golden trace
```

## Reading

- `AGENTS.md` — architecture, data flow, module map, every CLI flag
- `ARCHITECTURE.md`, `MASTER_PLAN.md`, `ROADMAP.md` — design review and the plan
- `bench/baseline.md` — measured stage timings
- `docs/blazehand-model-notes.md` — the BlazeHand ONNX export notes
