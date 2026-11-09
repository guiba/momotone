# Baseline measurements — Phase 0 (T0.1, T0.4)

Measured on this machine (M-series, macOS, rustc 1.98, `cargo build --release`)
after the telemetry rings landed. Every number is a **nearest-rank percentile** of a
128-sample ring, refreshed once a second, not a single frame's value.

Reproduce with:

```bash
make headless                          # live camera, MediaPipe
./target/release/gesture_rust --model yolo --score-threshold 0.3 --no-ui --max-frames 150 --debug
make mock
./target/release/gesture_rust --image output_frame.png --debug
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
