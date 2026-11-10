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
