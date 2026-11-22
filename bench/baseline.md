# Baseline measurements — Phase 0 (T0.1, T0.4)

Measured on this machine (M-series, macOS, rustc 1.98, `cargo build --release`)
after the telemetry rings landed. Every number is a **nearest-rank percentile** of a
128-sample ring, refreshed once a second, not a single frame's value.

Reproduce with:

```bash
make headless                          # live camera, MediaPipe
./target/release/momotone --model yolo --score-threshold 0.3 --no-ui --max-frames 150 --debug
make mock
./target/release/momotone --image output_frame.png --debug
```

---

## 1. Live camera, MediaPipe two-stage (`--no-ui --max-frames 300`, 640×480, requested 30 fps)

| Stage | p50 | p95 | p99 | max |
| :-- | :-- | :-- | :-- | :-- |
| Capture (camera read) | **33.2 ms** | 37.3 | 38.8 | 38.9 |
| Inference (tracker) | **8.2 ms** | 14.7 | 14.8 | 15.0 |
| camera → publish | 0.01 ms | 0.01 | 0.01 | 0.01 |
| dispatch → sink | 0.0 ms | 0.0 | 0.0 | 0.0 |

Loop: 300 frames in 9.99 s = **29.4 fps**, 33.30 ms/frame, **0 frames dropped**.
Audio stream: 44100 Hz, 2 channels.

Detector 5.5–8.2 ms; landmarker 4.8–5.8 ms *when it runs* (0 µs when the reuse cache
serves the frame).

**Reading:** the camera read *is* the loop. Inference is 25 % of it and never falls
behind, which is why nothing was dropped. The hand-to-event budget here is
capture 33.2 + inference 8.2 ≈ **41 ms**, before the audio block and the DAC.

## 2. Live camera, YOLO pose (`--model yolo --score-threshold 0.3 --max-frames 150`, 640×640)

| Stage | p50 | p95 | p99 | max |
| :-- | :-- | :-- | :-- | :-- |
| Capture | 33.7 ms | 41.9 | 45.9 | 47.3 |
| Inference | **0.0 ms** | 64.7 | 66.5 | 72.5 |
| camera → publish | 0.0 ms | 43.8 | 49.6 | 52.2 |

Loop: 150 frames in 6.47 s = 27.4 fps, 43.11 ms/frame, **45 of 195 frames dropped**.

Two things the averages hide:

* The **reuse cache short-circuits the whole pose model** when the palm is steady
  (3 px / 3 %, capped at 3 consecutive frames), so ~3 of every 4 frames run no model
  at all. That is why inference's p50 is 0 and its p95 is 65 ms. It is a 4× CPU win,
  but landmarks can be **up to 3 frames (≈100 ms) stale**.
* When inference does run (65 ms), it is slower than the camera, so frames queue and
  go **stale by up to 52 ms** before inference sees them — and those are dropped
  (counted in `dropped`).

## 3. Mock mode (`--mode mock --no-ui --max-frames 300`)

300 frames in 11.15 s = **26.4 fps**, 37.18 ms/frame against the 33 ms requested.
Confirms ROADMAP finding 1: `thread::sleep(33 ms)` overshoots on macOS, so every mock
telemetry figure (fps, trail decay, latency) is ~23 % off. Fix = deadline clock
(MASTER_PLAN T1.5).

Note: mock mode drains the event bus on the pipeline thread, so its dispatch→sink
ring stays empty by design.

## 4. Offline single image (`--image output_frame.png`)

```
box (47.9, 70.3, 454.6, 454.6)  score 0.965  landmarks 21
detector 6.07 ms   landmarker 4.51 ms
```

## 5. What this tells us for the plan

1. **Capture is the bottleneck, not inference** (33.2 ms vs 8.2 ms). T1.8 —
   capturing at 320×240 — is the single highest-leverage experiment left in the loop,
   because the detector only ever samples a 192×192 square.
2. **Detection flickers.** In the live run the confidence walked 0.97 → 0.67 → 0.00
   with a hand in frame, producing `silent` frames. Hysteresis (T1.7) is a
   correctness fix, not a nicety.
3. **Landmark jitter is real:** mean per-frame landmark displacement 4–15 px
   (MediaPipe), up to 46 px (YOLO). A cell is ~102 px wide, so the pitch is safe, but
   `spread`/`extension`/`tilt` are noisy — the estimator layer (T3.1) is justified.
4. **The sink thread is not a latency problem.** `recv_timeout(20 ms)` returns
   immediately when an event arrives, so dispatch→sink is sub-millisecond. The 20 ms
   is a *wakeup* cost (50 idle wakeups/s), not attack latency. `ARCHITECTURE.md`
   overstated it; corrected there.
5. **Unknown, and still to measure:** the audio block's own latency. cpal 0.17 does
   not expose the buffer size, so it has to be read from the callback's block length —
   that is a Phase 1 task.

---

# After Phase 1 (`phase1-hot-path`, same machine)

## Mock mode — deadline clock (T1.5)

300 frames in 9.88 s = **30.3 fps**, 32.92 ms/frame against the 33 ms requested
(was 26.4 fps / 37.18 ms). The clock is honest now, so every mock number can be
compared against the 33 ms budget.

## ONNX Runtime thread sweep (T1.6)

`--ort-threads N`, 200 frames, live camera, MediaPipe, no hand in view (so the
landmarker never runs and inference is the detector alone):

| `--ort-threads` | inference p50 | p95 | max |
| :-- | :-- | :-- | :-- |
| 1 | 13.7 ms | 23.7 | 25.4 |
| 2 | 11.7 ms | 16.6 | 22.4 |
| 4 | 10.8 ms | 12.9 | 25.4 |
| 0 (default, all cores) | 9.7 ms | 13.3 | 23.9 |

More threads are strictly faster for this model, and the loop is paced by the 33 ms
camera read anyway, so the flag defaults to 0 (runtime default) and exists for A/B
testing the jitter a full pool causes the UI/capture threads. Pinning to 2 is *not*
the win `ARCHITECTURE.md` assumed.

## Detection gate (T1.7)

2 confirm frames + 3 grace frames. In a 300-frame live run with no hand in front of
the camera the gate adds nothing measurable (capture 33.3 ms p50, inference 11.8 ms,
0 dropped, 30.3 fps), but it removes the on/off clicks the raw threshold produced:
the confidence sequence 0.97 → 0.67 → 0.00 with a hand in frame no longer emits a
release in the middle of a held gesture. Verified by the new pipeline test
`the_gate_needs_two_sightings_and_only_releases_after_the_grace_window`.

Still open: the audio block's own latency (cpal does not expose buffer size; read it
from the callback block length), and capture resolution (T1.8).

---

# After Phase 2 (`phase2-loop-rewrite`, same machine, release build)

Same commands as section 1, 300 frames, live camera, MediaPipe:

```
[Pipeline] 300 frames in 9.99 s: 29.9 fps, 33.31 ms/frame end to end (capture 35.26 ms, inference 11.54 ms, 0 frames dropped)
[Stages] p50/p95/p99/max ms - capture 33.5/38.4/39.1/40.4  inference 11.3/14.2/15.8/17.2  camera->publish 0.0/0.0/0.0/0.0  dispatch->sink 0.0/0.0/0.0/0.0  (128 samples)
```

| Metric | Phase 0/1 | Phase 2 |
| :-- | :-- | :-- |
| fps | 29.4 | 29.9 |
| ms/frame | 33.30 | 33.31 |
| capture p50 | 33.2 | 33.5 |
| inference p50 | 8.2 | 11.3 |
| dropped | 0 | 0 |

Mock mode (deadline clock): 300 frames in 9.87 s = **30.4 fps**, 32.90 ms/frame — the
clock is still honest after the loop rewrite.

With the dashboard open and the ASCII backdrop on (`--ascii --max-frames 250`): 29.7 fps,
capture 33.2 ms p50, inference 11.5 ms p50, **2 frames dropped** — the only frame loss in
the whole Phase 2 set, and it comes from the UI/ASCII work that still runs on the inference
thread. That is exactly the cost Phase 4 moves to a presenter thread.

The inference p50 difference vs section 1 (8.2 → 11.3 ms) is not the refactor: the Phase 0
run had no hand in front of the camera, so the 224×224 landmarker almost never ran. With a
hand in view the two-stage pipeline costs ~11 ms, and the loop is still paced by the 33 ms
camera read. **Net effect of Phase 2 on performance: none, by design** — the win is one
frame path (`next_frame` → `advance`) and six named stages instead of `run`/`tick` and a
210-line `advance`.

---

# After Phase 3 + Phase 4a (`phase3-instrument`, same machine, release build)

Live camera, MediaPipe, 300 frames, `--no-ui`:

```
[Pipeline] 300 frames in 9.99 s: 29.9 fps, 33.29 ms/frame end to end (capture 33.05 ms, inference 11.50 ms, 0 frames dropped)
[Stages] p50/p95/p99/max ms - capture 33.4/36.2/37.3/37.3  inference 10.2/12.5/13.2/15.0  camera->publish 0.0/0.0/0.0/0.0  dispatch->sink 0.0/0.0/0.0/0.0  (128 samples)
```

Live camera **with the ASCII backdrop on** (`--ascii --ascii-cols 70 --max-frames 300`):

```
[Pipeline] 300 frames in 9.99 s: 29.4 fps, 33.31 ms/frame end to end (capture 35.40 ms, inference 14.48 ms, 0 frames dropped)
[Stages] p50/p95/p99/max ms - capture 33.3/38.4/39.8/40.6  inference 11.7/12.9/13.8/14.5  camera->publish 0.0/0.0/0.0/0.0  dispatch->sink 0.0/0.0/0.0/0.0  (128 samples)
```

Mock mode: 300 frames in 9.87 s = **30.2 fps**, 32.91 ms/frame.

| Metric | Phase 2 | Phase 3 + 4a |
| :-- | :-- | :-- |
| fps (live, no UI) | 29.9 | 29.9 |
| capture p50 | 33.5 | 33.4 |
| inference p50 | 11.3 | 10.2 |
| dropped frames, UI + ASCII on | 2 | 0 (headless ASCII) |
| tests | 127 | **135** |

Two things to note:

1. Adding the ASCII render to a headless run costs ~1.5 ms of inference p50 (10.2 → 11.7)
   because the presenter still shares the process, but it no longer *delays a frame*: the
   frame thread only moves the `Mat` into a capacity-1 channel and drops it if the presenter
   is still drawing. Phase 2's UI+ASCII run dropped 2 frames from that same work.
2. The hand-to-sound budget is unchanged and still dominated by the camera read:
   capture 33.4 ms + inference 10.2 ms ≈ **43 ms** before the audio block and the DAC.

Still unmeasured: the audio callback's own block latency (cpal does not expose buffer size),
and capture resolution (T1.8).
---

# After Phase 5 infrastructure (`phase4-snapshot`, same machine, release build)

## Fixture clip run (`make fixture`, 300 frames, MediaPipe, no UI)

```
[Pipeline] 300 frames in 10.00 s: 29.7 fps, 33.34 ms/frame end to end (capture 0.23 ms, inference 12.82 ms, 0 frames dropped)
[Stages] p50/p95/p99/max ms - capture 0.2/0.2/0.3/0.3  inference 11.1/13.8/14.3/14.6  camera->publish 0.0/0.0/0.0/0.0  dispatch->sink 0.0/0.0/0.0/0.0  (128 samples)
```

Two things this run establishes:

1. **Decoding is not a cost.** A clip read from disk costs 0.2 ms per frame against the
   camera's 33 ms. The pacing is a deadline clock inside `VideoSource`, deliberately kept
   *outside* the timed capture region, so `capture_ms` measures the decode and not the wait.
   A fixture run therefore reports the same 33.34 ms/frame cadence as a live camera without
   pretending it did 33 ms of work.
2. **Inference is identical on the fixture.** 11.1 ms p50 vs 10.2–11.7 ms live, same
   detector/landmarker split, 0 dropped. The clip is a valid stand-in for a live run, which
   is what makes the golden trace meaningful.

## Golden trace (`make trace-check`)

```
[Trace] matches fixtures/clip_trace.json (120 frames)
```

Re-running the same clip against the recorded trace produces no diff, and the target exits
non-zero when a frame differs. `--trace <path>` writes a trace, `--expect <path>` compares
against one — so a change to the grid, the smoothing, the gate or the model that moves a
note shows up as a readable frame-by-frame diff instead of a screenshot.

## Tracker fixture test (`cargo test --release -- --include-ignored tracker_finds_the_hand_in_the_clip`)

120 frames of the clip, 106 detections (88 %), every detection carrying all 21 landmarks,
every reported box square with a palm centre inside the frame. This is the only test that
loads the real ONNX graphs; it is `#[ignore]`d because it needs the gitignored assets.

Tests: **146 passing** (6 ignored), gate green.

---

# After Phase 5 observability (`phase5-observability`, same machine, release build)

## Percentiles, measured rather than averaged

Three consecutive `make trace-check` runs of the 120-frame fixture on an idle machine,
showing the end-of-run `INFO` line that `--debug` now prints:

```
120 frames in 3.99 s: 29.8 fps, 33.25 ms/frame end to end (capture 0.22 ms, inference 11.52 ms, 0 frames dropped)
120 frames in 3.99 s: 29.8 fps, 33.25 ms/frame end to end (capture 0.22 ms, inference 11.53 ms, 0 frames dropped)
120 frames in 3.99 s: 29.8 fps, 33.25 ms/frame end to end (capture 0.22 ms, inference 11.78 ms, 0 frames dropped)
```

Frame-to-frame the run is stable to within 0.26 ms of inference p50, and the trace is
byte-identical each time. The percentiles the Performance card now shows come from the
same rings; nothing new is measured, only what was already being thrown away.

Live camera, 60 frames, `make headless`:

```
capture   p50 33.2  p95 56.7  p99 62.7  max 62.7
inference p50 12.4  p95 17.6  p99 20.0  max 20.0
camera->publish 0.0 / 0.1 / 5.9 / 5.9
dispatch->sink  0.0 / 0.0 / 0.0 / 0.0
```

**Reading:** capture's p95 (56.7 ms) is 70 % above its median — the camera driver is the
unreliable stage, not inference. Inference's whole distribution fits inside one frame
interval with room to spare. This is the number the budget guard keys on, and it is why
the guard looks at p95 rather than the mean: an average of 12 ms hides a 57 ms tail.

## The subsystem budget (`make budget`)

`--budget-ms 5` against the fixture, where inference p95 is 19.4 ms:

```
WARN momotone::pipeline: inference p95 19.4 ms exceeds the 5 ms budget - ascii and camera preview off
```

At the shipped default (`--budget-ms 33`, the 30 fps frame interval) the guard does **not**
fire on either backend: MediaPipe inference p95 is 11.7–17.6 ms, and even the fixture's
worst tail is under 33 ms. So the guard is dormant in normal use — it is a load-shedding
policy that only bites when something else (the pose model, a second process, a 4K
capture) pushes inference past the interval. `--budget-ms 0` disables it entirely.

Tests: **152 passing** (6 ignored), `make verify` green.

---

# After OpenCV removal (`chore/opencv-removal`, same machine, rustc 1.100)

## Build cost — the headline number

| | with `opencv` | without |
| :-- | --: | --: |
| Cold `cargo check --all-targets` | **19 m 12 s** | **34 s** |
| Cold `cargo build --release` | ~20 min | **3 m 52 s** |
| Warm `cargo check` | ~4 s | ~0.3 s |

The OpenCV crate's build script runs `clang` over the whole OpenCV header tree to
generate the bindings; that was the single largest item in a cold build. Capture,
decoding, sampling, the ASCII downsample and the `--image` overlay are now pure Rust,
so there is no native toolchain requirement at all: `.cargo/config.toml`, which pinned
`LIBCLANG_PATH`/`DYLD_LIBRARY_PATH` to Homebrew LLVM, is gone.

`build.rs` links clang's compiler-rt on Apple targets. rustc 1.100's fat LTO drops
`___isPlatformVersionAtLeast` from `std` (rustc issue #164915), and ONNX Runtime's
CoreML object references it, so a release build with `lto = "fat"` fails to link
without it. LTO is worth keeping: the Rust hot path is ~13 % faster with it.

## Hot-path stages (`cargo test --release -- --include-ignored --nocapture`)

| Stage | OpenCV | pure Rust |
| :-- | --: | --: |
| Detector sampling 192x192 | 133 µs | 146–180 µs |
| Landmarker sampling 224x224 | 94 µs | 254–278 µs |
| ASCII downsample 640x480 -> 140x105 | ~0.3 ms (`imgproc::resize`) | **0.30 ms** (hand-written box average) |
| `Frame::to_rgba` 640x480 | 0.30 ms | 0.32 ms |
| MP4/H.264 decode + RGB convert, 640x480 | ~0.2 ms (OpenCV `VideoCapture`) | **1.17–1.25 ms** (`mp4io` + `rusty_h264-decoder`) |
| Webcam YUYV -> RGB, 640x480 | (OpenCV decoded it) | **1.4 ms** hand-written; nokhwa's own RGBA decoder is **53 ms** |

The sampler is the one place the hand-written code is slower than OpenCV's `resize`:
OpenCV vectorises the bilinear tap, our loop does not, so the 224x224 landmarker crop
costs ~2.5x more. On a 33 ms frame budget that is noise — the whole capture stage of a
fixture run is 1.9 ms p50.

## Detector window: anchored crop vs whole frame (PR 2)

`make probe` runs `--detector-window both`, which keeps the anchored crop as the
behaviour and scores the letterboxed whole frame alongside it. Two windows, one run,
identical conditions. 200 frames each, release build, quiet machine.

| clip | window | frames with a hand | mean best score | hands/frame | frames with 2+ hands | detector µs/frame |
| :-- | :-- | --: | --: | --: | --: | --: |
| `hand_clip.mp4` (640×480, one hand) | crop | 173 | 0.838 | 0.87 | 1 | 8126 |
| | frame | 162 | 0.785 | 0.81 | **0** | 7902 |
| `2_hands_640.mp4` (640×480) | crop | 200 | 0.953 | 1.46 | 92 | 10486 |
| | frame | 200 | 0.957 | **1.90** | **179** | 10181 |
| `2_hands_clip.mp4` (1620×1080) | crop | 199 | 0.951 | 1.01 | 4 | 8900 |
| | frame | 198 | 0.949 | **1.84** | **171** | 8436 |

Three things fall out of this:

- **The whole-frame window is not slower.** Both windows sample the same 192×192
  buffer; only the affine differs. 7.9 ms vs 8.1 ms is within run-to-run noise. A
  previous reading of "the frame window is 2.5× slower" was contention from a second
  copy of the app running in another terminal.
- **Only the frame window sees a second hand.** At native 1620×1080 the anchored crop -
  a `min(W,H)` = 1080 square - finds two palms in 4 frames out of 200, because the two
  hands are further apart than the window can span. The letterbox finds them in 171.
- **The frame window does lose detections on a single distant hand** (173 → 162), and
  it is not a gate problem: sweeping `--score-threshold` 0.6 → 0.5 → 0.4 moves the
  frame window 162 → 165 → 169 while the crop goes 173 → 178 → 179 and its false
  positives grow from 1 frame to 11. Lowering the threshold buys crop noise, not frame
  recall.

Across 753 detections on the two-hand clips, no palm centre landed in the top 22% of
the frame where the face and torso are: the letterbox padding does not manufacture
hands.

### The separation gate

Hands per frame on `2_hands_640.mp4` as the NMS gate widens:

| gate | 0.25 | 0.5 | 1.0 | 2.0 |
| :-- | --: | --: | --: | --: |
| crop | 1.46 | 1.46 | 1.46 | 1.18 |
| frame | 1.90 | 1.90 | 1.90 | 1.34 |

One hand's duplicate anchors merge at any gate up to 1.0; separate hands only start
merging at 2.0. Measured palm-centre separation in the clip is 1.3-2.6 palm radii.
`HAND_SEPARATION = 1.0` is the widest gate that still separates the hands while leaving
margin before they merge.

## Fixture and trace

`fixtures/hand_clip.mp4` is a recording of a real hand, so it is **not in the repo**
(`fixtures/*.mp4` and `fixtures/clip_trace.json` are gitignored). `make record` generates it
locally; `make fixture`, `make trace-check` and `make trace` use it. It had to be re-encoded
to `yuv420p`: `rusty_h264-decoder` is 4:2:0-only and the original clip was High 4:2:2. The
decode is **bit-exact against ffmpeg's YUV planes** over all 599 frames after reordering
B-frames into display order; the RGB conversion lands within 2 LSB of ffmpeg's `rgb24`.

Because the fixture's pixels changed, the golden trace changed with it and was regenerated:

| | old trace | new trace |
| :-- | :-- | :-- |
| First detection | frame 17 | frame 2 |
| First note / volume | 62 / 0.39 | 64 / 0.37 |
| Detections in 120 frames | 104 | 94 |

`make trace-check` passes, and two consecutive runs write byte-identical traces.

## The trace CI can actually run

There is no macOS runner, so CI can never open a camera or load the gitignored ONNX assets.
Mock mode is driven by the **frame index**, so 120 mock frames always produce the same hand
at the same place — `fixtures/mock_trace.json` is committed and CI diffs against it
(`make mock-trace-check`). It covers the detection gate, the estimator, the grid strategy
and the release path, and it is what catches a change to the gesture mapping on Linux.

```bash
make mock-trace          # rewrite fixtures/mock_trace.json
make mock-trace-check    # CI's gate
```

## End to end

```
# fixture, MediaPipe, 300 frames
300 frames in 10.00 s: 29.7 fps, 33.35 ms/frame (capture 1.9 ms, inference 12.9 ms, 0 dropped)

# live webcam, nokhwa, 60 frames
60 frames in 2.53 s: 28.5 fps, 42.20 ms/frame (capture 33.1 ms, inference 9.6 ms, 0 dropped)

# YOLO pose on the fixture, 30 frames
30 frames in 0.99 s: 32.1 fps (capture 2.7 ms, inference 22.7 ms p50)
```

Tests: **181 passing** (7 ignored), `make verify` green, `unsafe` down to the single
documented `unsafe impl Send for StreamWrapper` (four raw-pointer reads in the tracker's
tensor sampling went away with OpenCV).

## The effect chain costs nanoseconds, not milliseconds

The chain (one-pole lowpass, LFO vibrato, comb delay) runs inside the cpal callback, so the
number that matters is cost per audio sample, not per frame. Measured on the same machine,
`cargo test --release -- --include-ignored bench_chain_cost --nocapture` (1 M samples at
44.1 kHz):

| Voice | ns/sample | Cost over one 33 ms frame |
| :-- | :-- | :-- |
| Sine, no chain | 5.7 | 0.19 ms |
| Sine, all three stages | 12.8 | 0.42 ms |
| Square, no chain | 31.6 | 1.04 ms |
| Square, all three stages | 36.9 | 1.22 ms |

The chain adds ~7 ns per sample, i.e. ~0.2 ms per frame - 0.6 % of the 33 ms budget, and it
does not touch inference at all (the parameters are computed once per frame on the pipeline
thread and read once per audio block). The timbre is an order of magnitude more expensive
than the filtering: a band-limited square is 5x the cost of a sine before any effect is
switched on.

The delay ring is `DELAY_MAX_MS` (600 ms) long, allocated once in `Voice::idle` when the
stream opens: ~106 KB at 44.1 kHz, never reallocated by the callback.

